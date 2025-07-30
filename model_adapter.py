import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Any, Optional, Callable
import numpy as np
from PIL import Image
import torch
import dtlpy as dl
from dtlpyconverters import services, coco_converters
from dtlpy.services import service_defaults
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.config import TrainConfig

logger = logging.getLogger('rf-detr-adapter')


class ModelAdapter(dl.BaseModelAdapter):

    @staticmethod
    def _process_coco_json(output_annotations_path: str) -> None:
        """
        Process COCO JSON annotations file to make it compatible with RF-DETR requirements.

        RF-DETR requires integer IDs and supercategory fields for categories. This function converts
        string IDs to integers via hashing, adds missing supercategory fields, and updates file paths
        to match new image locations by keeping only filenames.

        Args:
            output_annotations_path (str): Path to directory containing the COCO JSON file

        Returns:
            None
        """
        src_json_path = os.path.join(output_annotations_path, 'coco.json')
        dest_json_path = os.path.join(output_annotations_path, '_annotations.coco.json')

        logger.info(f'Processing COCO JSON file at {src_json_path}')
        # Load the JSON file
        with open(src_json_path, 'r') as f:
            coco_data = json.load(f)

        # Add supercategory field to each category if it doesn't exist
        for category in coco_data.get('categories', []):
            if 'supercategory' not in category:
                category['supercategory'] = 'none'

        # Convert image IDs to integers and clean file names
        for image in coco_data.get('images', []):
            if isinstance(image['id'], str):
                image['id'] = abs(hash(image['id']))
            # Update file_name to match new filename format with path information
            if '/' in image['file_name']:
                # Convert path separators to underscores and prepend relative path
                relative_path = os.path.dirname(image['file_name'])
                filename = os.path.basename(image['file_name'])
                if relative_path == ".":
                    image['file_name'] = filename  # Keep original filename for root files
                else:
                    image['file_name'] = f"{relative_path.replace('/', '_')}_{filename}"

        # Convert annotation IDs and image_ids to integers
        for annotation in coco_data.get('annotations', []):
            if isinstance(annotation['id'], str):
                annotation['id'] = abs(hash(annotation['id']))
            if isinstance(annotation['image_id'], str):
                annotation['image_id'] = abs(hash(annotation['image_id']))

        with open(dest_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info('COCO JSON processing completed')

    @staticmethod
    def _extract_yolo_like_metrics(rf_detr_metrics: dict) -> dict:
        """
        Extract YOLO-like metrics from RF-DETR training metrics.

        This method processes the metrics dictionary received from RF-DETR training and converts
        it into a format similar to YOLO metrics for consistency and comparison purposes.

        The following metrics are extracted:
        - box_loss: Combined bbox regression loss and GIoU loss
        - cls_loss: Classification loss from train_loss_ce
        - mAP50-95(B): Mean Average Precision across IoU thresholds [0.50:0.95]
        - mAP50(B): Average Precision at IoU=0.50
        - recall(B): Average Recall with 100 detections per image

        Args:
            rf_detr_metrics (dict): Dictionary containing RF-DETR training metrics

        Returns:
            dict: Dictionary containing YOLO-like metrics extracted from RF-DETR data
        """
        result = {}

        # Calculate box_loss: sum of bbox regression loss and GIoU loss
        bbox_loss = rf_detr_metrics.get("train_loss_bbox")
        giou_loss = rf_detr_metrics.get("train_loss_giou")
        if bbox_loss is not None and giou_loss is not None:
            result["box_loss"] = bbox_loss + giou_loss

        # Classification loss (cls_loss): directly mapped from train_loss_ce
        cls_loss = rf_detr_metrics.get("train_loss_ce")
        if cls_loss is not None:
            result["cls_loss"] = cls_loss

        # Extract COCO evaluation metrics (from ema_test_coco_eval_bbox)
        # COCO eval returns a list of metrics:
        # [0] AP@[.50:.95] (mean Average Precision across IoU thresholds)
        # [1] AP@0.50 (average precision at IoU=0.5)
        # [8] AR@100 (average recall with 100 detections per image)
        coco_eval = rf_detr_metrics.get("ema_test_coco_eval_bbox")
        if coco_eval and isinstance(coco_eval, list) and len(coco_eval) >= 9:
            result["mAP50-95(B)"] = coco_eval[0]  # COCO's AP@[.50:.95], matches YOLO's mAP50-95(B)
            result["mAP50(B)"] = coco_eval[1]  # COCO's AP@0.50, matches YOLO's mAP50(B)
            result["recall(B)"] = coco_eval[8]  # COCO's AR@100, used as recall(B) proxy
        else:
            # If COCO metrics are missing or incomplete, fill with None
            result["mAP50-95(B)"] = None
            result["mAP50(B)"] = None
            result["recall(B)"] = None

        return result

    def _get_rf_detr_train_config(self, data_path: str, output_path: str) -> TrainConfig:
        """
        Get RF-DETR training configuration from model configuration.

        This method creates a TrainConfig object with parameters from the model's configuration.
        It sets up training hyperparameters like learning rates, batch size, epochs etc.
        The configuration includes:
        - Dataset and output paths
        - Training parameters (epochs, batch size, learning rates)
        - Optimization settings (gradient accumulation, weight decay)
        - Model checkpointing and early stopping parameters
        - Class names from the model's label map

        Args:
            data_path (str): Path to directory containing the training data
            output_path (str): Path where model outputs and checkpoints will be saved

        Returns:
            TrainConfig: Configuration object for RF-DETR training with all parameters set

        Note:
            Default values are used for parameters not specified in the model configuration.
            The number of workers is set to 0 to avoid multiprocessing issues.
        """
        train_config_dict = self.configuration.get('train_configs', {})
        class_names = None
        if self.model_entity.dataset.instance_map is not None:
            class_names = list(self.model_entity.dataset.instance_map.keys())

        # Initialize with required parameters
        return TrainConfig(
            dataset_dir=data_path,
            output_dir=output_path,
            num_workers=0,  # default num_workers cause issue in loading data
            epochs=train_config_dict.get('epochs', 10),
            batch_size=train_config_dict.get('batch_size', 4),
            grad_accum_steps=train_config_dict.get('grad_accum_steps', 4),
            lr=train_config_dict.get('lr', 1e-4),
            lr_encoder=train_config_dict.get('lr_encoder', 1.5e-4),
            weight_decay=train_config_dict.get('weight_decay', 1e-4),
            use_ema=train_config_dict.get('use_ema', True),
            checkpoint_interval=train_config_dict.get('checkpoint_interval', 10),
            early_stopping_patience=train_config_dict.get('early_stopping_patience', 10),
            early_stopping_min_delta=train_config_dict.get('early_stopping_min_delta', 0.001),
            early_stopping_use_ema=train_config_dict.get('early_stopping_use_ema', False),
            class_names=class_names,
        )

    def on_epoch_end(self, data: dict, output_path: str, faas_callback: Optional[Callable] = None) -> None:
        """
        Callback executed at the end of each training epoch.

        This method processes epoch metrics, updates model state, and saves metrics to the model entity.
        It handles:
        - Updating current epoch counter
        - Executing FaaS callback if provided to report progress
        - Saving metrics to the model entity
        - Updating the start_epoch configuration for next epoch

        Args:
            data (dict): Dictionary containing epoch training data and metrics
            output_path (str): Path where model outputs and checkpoints will be saved
            faas_callback (Optional[Callable]): Optional callback function to report progress,
                                              takes current epoch and total epochs as arguments

        Returns:
            None
        """
        self.current_epoch = data['epoch']
        if faas_callback is not None:
            faas_callback(self.current_epoch, self.train_config.epochs)
        samples = list()
        NaN_dict = {'box_loss': 1, 'cls_loss': 1, 'mAP50(B)': 0, 'mAP50-95(B)': 0, 'recall(B)': 0}

        yolo_metrics = ModelAdapter._extract_yolo_like_metrics(data)
        for metric_name, value in yolo_metrics.items():
            if not np.isfinite(value):
                filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                filters.add(field='modelId', values=self.model_entity.id)
                filters.add(field='figure', values=metric_name)
                filters.add(field='data.x', values=self.current_epoch - 1)
                items = self.model_entity.metrics.list(filters=filters)

                if items.items_count > 0:
                    value = items.items[0].y
                else:
                    value = NaN_dict.get(metric_name, 0)
                logger.warning(f'Value is not finite. For figure {metric_name} and legend metrics using value {value}')
            samples.append(dl.PlotSample(figure=metric_name, legend='metrics', x=self.current_epoch, y=value))
        self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)

        self.configuration['start_epoch'] = self.current_epoch + 1
        logger.info(f'Saving model from {output_path}')
        self.save_to_model(local_path=output_path, cleanup=False)

    def save(self, local_path: str, **kwargs) -> None:
        """
        Save model configuration by updating the weights filename.

        This method updates the model configuration to point to the best checkpoint weights file.

        Args:
            local_path (str): Path where model files are saved (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            None
        """
        self.configuration.update({'weights_filename': 'checkpoint_best_total.pth'})

    def load(self, local_path: str, **kwargs) -> None:
        """
        Load the model weights and configurations.

        This method searches for model weights first in the specified local path,
        then in /tmp/app/weights directory. If weights are not found in either location,
        default pretrained weights will be used.

        The weights in /tmp/app/weights are downloaded in docker image

        Args:
            local_path (str): Directory path containing the model files
            **kwargs: Additional keyword arguments (unused)

        Returns:
            None
        """
        logger.info(f'Loading model from {local_path}')

        model_filename = self.configuration.get('weights_filename', 'rf-detr-base-coco.pth')
        local_model_filepath = os.path.normpath(os.path.join(local_path, model_filename))
        default_weights = os.path.join('/tmp/app/weights', model_filename)

        # when weights_path is None, the model will be loaded from the default weights
        if os.path.isfile(local_model_filepath):
            weights_path = local_model_filepath
        elif os.path.isfile(default_weights):
            weights_path = default_weights
        else:
            weights_path = None

        use_rf_detr_large = self.configuration.get('use_rf_detr_large', False)
        if model_filename == 'rf-detr-large.pth':
            use_rf_detr_large = True

        self.confidence_threshold = self.configuration.get('conf_thres', 0.25)
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Get the number of classes from the model entity
        num_classes = len(self.model_entity.labels)

        logger.info(
            f'Loading model with weights: {weights_path}, '
            f'confidence threshold: {self.confidence_threshold}, '
            f'device: {device_name}, '
            f'num_classes: {num_classes}, '
            f'use_large_model: {use_rf_detr_large}'
        )
        if use_rf_detr_large is False:
            self.model = RFDETRBase(
                pretrain_weights=weights_path,
                device=device_name,
                num_classes=num_classes,  # Pass the correct number of classes
            )
        else:
            self.model = RFDETRLarge(
                pretrain_weights=weights_path,
                device=device_name,
                num_classes=num_classes,  # Pass the correct number of classes
            )

    def prepare_item_func(self, item):
        if 'image' not in item.mimetype:
            raise Exception('RF-DETR only supports images')
        buffer = item.download(save_locally=False)
        image = np.asarray(Image.open(buffer).convert('RGB'))
        return image

    def predict(self, batch: List[Any], **kwargs) -> List[dl.AnnotationCollection]:
        """Run predictions on a batch of data.

        Args:
            batch (List[Any]): List of images to run prediction on
            **kwargs: Additional keyword arguments (unused)

        Returns:
            List[dl.AnnotationCollection]: List of annotation collections, one per image,
                containing detected objects as boxes with labels and confidence scores
        """
        logger.info(f'Predicting batch of size: {len(batch)}')
        results = self.model.predict(batch, threshold=self.confidence_threshold)

        batch_annotations = []
        # model.predicts returns a list if batch is a list but a single object if batch is a single object
        if not isinstance(results, list):
            results = [results]

        for i, detection in enumerate(results):
            image_annotations = dl.AnnotationCollection()
            
            # Get image dimensions
            image = batch[i]
            img_height, img_width = image.shape[:2]
            
            for xyxy, class_id, conf in zip(detection.xyxy, detection.class_id, detection.confidence):
                label = self.model.class_names[class_id]
                
                # Check and fix boundary issues
                x1, y1, x2, y2 = xyxy
                
                # Clamp coordinates to image boundaries
                x1_fixed = max(0, min(x1, img_width))
                y1_fixed = max(0, min(y1, img_height))
                x2_fixed = max(0, min(x2, img_width))
                y2_fixed = max(0, min(y2, img_height))
                
                # Warn if coordinates were outside boundaries
                if (x1 != x1_fixed or y1 != y1_fixed or x2 != x2_fixed or y2 != y2_fixed):
                    logger.warning(f'Bounding box {xyxy} outside image boundaries ({img_width}x{img_height}). Fixed to {x1_fixed, y1_fixed, x2_fixed, y2_fixed}')
                
                image_annotations.add(
                    dl.Box(left=x1_fixed, top=y1_fixed, right=x2_fixed, bottom=y2_fixed, label=label),
                    model_info={'name': self.model_entity.name, 'model_id': self.model_entity.id, 'confidence': conf},
                )
            batch_annotations.append(image_annotations)
        return batch_annotations

    def convert_from_dtlpy(self, data_path: str, **kwargs) -> None:
        """Convert dataset from Dataloop format to COCO format.

        This method converts a Dataloop dataset to COCO format required by RF-DETR. It validates box annotations
        in each subset (train/validation) and converts them to match RF-DETR's train/valid directory structure.

        Args:
            data_path (str): Path to the directory where the dataset will be converted
            **kwargs: Additional keyword arguments (unused)

        Raises:
            ValueError: If model has no labels defined or if no box annotations are found in a subset
        """
        logger.info(f'Converting dataset from Dataloop format to COCO format at {data_path}')

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if len(self.model_entity.labels) == 0:
            logger.error("Model has no labels defined")
            raise ValueError('model.labels is empty. Model entity must have labels')

        for subset_name in subsets.keys():
            logger.info(f'Converting subset: {subset_name} to COCO format')

            # rf-detr expects train and valid folders
            dist_dir_name = subset_name if subset_name != 'validation' else 'valid'
            input_annotations_path = os.path.join(data_path, subset_name, 'json')
            output_annotations_path = os.path.join(data_path, dist_dir_name)

            # Ensure instance map IDs start from 1 not 0
            if 0 in self.model_entity.dataset.instance_map.values():
                self.model_entity.dataset.instance_map = {
                    label: label_id + 1 for label, label_id in self.model_entity.dataset.instance_map.items()
                }

            converter = coco_converters.DataloopToCoco(
                output_annotations_path=output_annotations_path,
                input_annotations_path=input_annotations_path,
                download_items=False,
                download_annotations=False,
                dataset=self.model_entity.dataset,
            )

            coco_converter_services = services.converters_service.DataloopConverters()
            loop = coco_converter_services._get_event_loop()
            try:
                loop.run_until_complete(converter.convert_dataset())
            except Exception as e:
                raise Exception(f"Error converting subset {subset_name}: {str(e)}")

            # convert coco.json to _annotations.coco.json
            ModelAdapter._process_coco_json(output_annotations_path)

            # Move images from the deepest subdirectory under <data_path>/<subset_name>/items/
            # to <data_path>/<dist_dir_name>. For example:
            # From: <data_path>/train/items/dir1/dir2/images.jpg
            # To:   <data_path>/train/images.jpg
            src_parent_dir = os.path.join(data_path, subset_name, 'items')
            dst_images_path = os.path.join(data_path, dist_dir_name)

            # Moving files to a temporary directory first because in most cases source and destination
            # directories have the same name (e.g. 'train' -> 'train'). Direct move would fail.
            tmp_dir_path = os.path.join(data_path, f'tmp_dir_{subset_name}')
            os.makedirs(tmp_dir_path)
            json_file_path = os.path.join(output_annotations_path, '_annotations.coco.json')
            shutil.move(json_file_path, tmp_dir_path)

            # move all non json files to tmp_dir_path
            all_files = [f for f in Path(src_parent_dir).rglob('*') if f.is_file() and not f.name.endswith('.json')]
            if len(all_files) == 0:
                raise FileNotFoundError(f"No files found in {src_parent_dir}")
            for file in all_files:
                # Get the relative path from the source directory
                relative_path = os.path.relpath(file.parent, src_parent_dir)
                if relative_path == ".":
                    new_filename = file.name  # Keep the original filename for root files
                else:
                    new_filename = f"{relative_path.replace(os.sep, '_')}_{file.name}"
                # Create new file path in tmp_dir with unique filename
                new_file_path = os.path.join(tmp_dir_path, new_filename)
                shutil.move(file, new_file_path)

            # Remove the original subset directory
            logger.info(f'Removing {os.path.join(data_path, subset_name)}')
            shutil.rmtree(os.path.join(data_path, subset_name))
            if os.path.exists(dst_images_path):
                shutil.rmtree(os.path.join(dst_images_path))

            logger.info(f'Moving directory from {tmp_dir_path} to {dst_images_path}')
            shutil.move(tmp_dir_path, dst_images_path)


    def train(self, data_path: str, output_path: str, **kwargs) -> None:
        """
        Train the RF-DETR model on the provided dataset.

        Steps:
        1. Get training configuration from model configuration
        2. Handle resuming from checkpoint if start_epoch > 0
        3. Add FaaS callback for epoch end events
        4. Train the model
        5. Update start_epoch in configuration

        Args:
            data_path (str): Path to directory containing the training data in COCO format
            output_path (str): Path where trained model checkpoints and outputs will be saved
            **kwargs: Additional keyword arguments
                on_epoch_end_callback (Callable): Optional callback function to execute at the end of each epoch

        Returns:
            None

        Raises:
            RuntimeError: If CUDA device does not support bfloat16 dtype
        """
        logger.info(f'Starting training with data from {data_path}')

        self.train_config = self._get_rf_detr_train_config(data_path, output_path)
        logger.info(f'train_config: {self.train_config}')

        start_epoch = self.configuration.get('start_epoch', 0)
        # Find the most recent checkpoint file to resume training from if start_epoch > 0
        resume_checkpoint = ''
        if start_epoch > 0:
            checkpoint_path = os.path.join(
                service_defaults.DATALOOP_PATH, "models", self.model_entity.name, 'checkpoint.pth'
            )
            if not os.path.isfile(checkpoint_path):
                raise Exception(f'No checkpoint found at {checkpoint_path}')
            else:
                logger.info(f'resume from checkpoint: {resume_checkpoint}')
                resume_checkpoint = checkpoint_path

        # Add callback for epoch end events
        self.model.callbacks["on_fit_epoch_end"].append(
            lambda data: self.on_epoch_end(data, output_path, kwargs.get('on_epoch_end_callback'))
        )

        training_kwargs = {}
	# torch.cuda.is_bf16_supported will crash if cuda isnt supported
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            logger.warning("CUDA device does not support bfloat16. Using fp16_eval=False and amp=False")
            training_kwargs = {'fp16_eval': False, 'amp': False}

        logger.info('start rf-detr training')
        self.model.train_from_config(config=self.train_config, resume=resume_checkpoint, **training_kwargs)

        #  Check if the model (checkpoint) has already completed training for the specified number of epochs, if so, can start again without resuming
        if 'start_epoch' in self.configuration and self.configuration['start_epoch'] == self.train_config.epochs:
            self.configuration['start_epoch'] = 0
            self.model_entity.update()

        logger.info('Training completed')

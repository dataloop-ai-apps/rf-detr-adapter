import json
import sys
import logging
import os
import glob
import shutil
from typing import List, Any
import numpy as np
import torch
import dtlpy as dl
from dtlpyconverters import services, coco_converters

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

logger = logging.getLogger('rf-detr-adapter')


class ModelAdapter(dl.BaseModelAdapter):

    @staticmethod
    def _copy_files(src_path: str, dst_path: str) -> None:
        logger.info(f'Copying files from {src_path} to {dst_path}')
        print(f'-HHH- src_path: {src_path}')
        print(f'-HHH- dst_path: {dst_path}')
        os.makedirs(dst_path, exist_ok=True)
        for filename in os.listdir(src_path):
            file_path = os.path.join(src_path, filename)
            print(f'-HHH- file_path: {file_path}')
            if os.path.isfile(file_path):
                new_file_path = os.path.join(dst_path, filename)
                print(f'-HHH- new_file_path: {new_file_path}')
                shutil.copy(file_path, new_file_path)
        logger.info('File copy completed')

    @staticmethod
    def _process_coco_json(output_annotations_path: str) -> None:
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
            # Remove parent directory from file_name
            if '/' in image['file_name']:
                image['file_name'] = os.path.basename(image['file_name'])

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

    def on_epoch_end(self, data, faas_callback=None):
        # get last epoch checkpoint
        self.current_epoch = data['epoch']
        if faas_callback is not None:
            faas_callback(self.current_epoch, self.configuration.get('epochs', 10))
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
            samples.append(dl.PlotSample(figure=metric_name, legend='matrics', x=self.current_epoch, y=value))
        self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)

        self.configuration['start_epoch'] = self.current_epoch + 1
        # TODO : check if that is needed ? since i see its already called in BaseModelAdapter
        # self.save_to_model(local_path=self.configuration.get('output_path', ''), cleanup=False)

    def save(self, local_path: str, **kwargs) -> None:
        self.configuration.update({'weights_filename': 'checkpoint_best_total.pth'})

    def load(self, local_path: str, **kwargs) -> None:
        """Load your model from saved weights"""
        logger.info(f'Loading model from {local_path}')

        model_filename = self.configuration.get('weights_filename', 'rf-detr-base-coco.pth')
        logger.info(f'-HHH- model_filename: {model_filename}')
        model_filepath = os.path.normpath(os.path.join(local_path, model_filename))
        logger.info(f'-HHH- model_filepath: {model_filepath}')
        default_weights = os.path.join('/tmp/app/weights', model_filename)

        # when weights_path is None, the model will be loaded from the default weights
        weights_path = None
        if os.path.isfile(model_filepath):
            weights_path = model_filepath
        elif os.path.isfile(default_weights):
            weights_path = default_weights

        logger.info(f'-HHH- weights_path: {weights_path}')
        self.confidence_threshold = self.configuration.get('conf_thres', 0.25)

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get the number of classes from the model entity
        num_classes = len(self.model_entity.labels)
        logger.info(f'Number of classes in dataset: {num_classes}')

        logger.info(
            f'Loading model with weights: {weights_path}, '
            f'confidence threshold: {self.confidence_threshold}, '
            f'device: {device_name}, '
            f'num_classes: {num_classes}'
        )

        self.model = RFDETRBase(
            pretrain_weights=weights_path,
            device=device_name,
            num_classes=num_classes,  # Pass the correct number of classes
        )
        logger.info(f'-HHH- model created')

    # rf-detr is resize, normalize and convert to tensor in the model
    # nothing to do here
    # def prepare_item_func(self, item):
    #     pass

    def predict(self, batch: List[Any], **kwargs) -> List[dl.AnnotationCollection]:
        """Run predictions on a batch of data"""
        logger.info(f'Predicting batch of size: {len(batch)}')
        results = self.model.predict(batch, threshold=self.confidence_threshold)

        batch_annotations = []
        # model.predicts returns a list if batch is a list but a single object if batch is a single object
        if not isinstance(results, list):
            results = [results]

        for detection in results:
            image_annotations = dl.AnnotationCollection()
            for xyxy, class_id, conf in zip(detection.xyxy, detection.class_id, detection.confidence):
                label = COCO_CLASSES[class_id]
                image_annotations.add(
                    dl.Box(left=xyxy[0], top=xyxy[1], right=xyxy[2], bottom=xyxy[3], label=label),
                    model_info={
                        'name': self.model_entity.name,
                        'model_id': self.model_entity.id,
                        'confidence': conf,
                        'dataset_id': self.model_entity.dataset_id,
                    },
                )
            batch_annotations.append(image_annotations)
        return batch_annotations

    def convert_from_dtlpy(self, data_path: str, **kwargs) -> None:
        logger.info(f'Converting dataset from Dataloop format at {data_path}')

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        print(f'-HHH- subsets: {subsets}')

        if len(self.model_entity.labels) == 0:
            logger.error("Model has no labels defined")
            raise ValueError('model.labels is empty. Model entity must have labels')

        # TODO : learn how this code works ( add debug messages)
        for subset, filters_dict in subsets.items():
            logger.info(f'Processing subset: {subset}')
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values=['box'], operator=dl.FILTERS_OPERATIONS_IN)
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                logger.error(f"No box annotations found in subset: {subset}")
                raise ValueError(
                    f'Could not find box annotations in subset {subset}. Cannot train without annotations in the data subsets'
                )

        # TODO: check if that is needed
        self.model_entity.dataset.instance_map = self.model_entity.label_to_id_map

        for subset_name in subsets.keys():
            logger.info(f'Converting subset: {subset_name} to COCO format')
            dist_dir_name = subset_name if subset_name != 'validation' else 'valid'
            input_annotations_path = os.path.join(data_path, subset_name, 'json')
            output_annotations_path = os.path.join(data_path, dist_dir_name)

            converter = coco_converters.DataloopToCoco(
                output_annotations_path=output_annotations_path,
                input_annotations_path=input_annotations_path,
                download_items=False,
                download_annotations=False,
                dataset=self.model_entity.dataset,
                filters=dl.Filters(custom_filter=subsets[subset_name]),
            )

            coco_converter_services = services.converters_service.DataloopConverters()
            loop = coco_converter_services._get_event_loop()
            try:
                loop.run_until_complete(converter.convert_dataset())
            except Exception as e:
                logger.error(f"Error converting subset {subset_name}: {str(e)}")
                raise

            ModelAdapter._process_coco_json(output_annotations_path)

            src_images_path = os.path.join(data_path, subset_name, 'items', subset_name)
            dst_images_path = os.path.join(data_path, dist_dir_name)
            ModelAdapter._copy_files(src_images_path, dst_images_path)

    def train(self, data_path: str, output_path: str, **kwargs) -> None:
        logger.info(f'Starting training with data from {data_path}')

        logger.info(f'-HHH- ver 28-apr data_path: {data_path}')
        print(f"-HHH- ver 28-apr data_path: {data_path}")
        train_config = self.configuration.get('train_configs', {})
        epochs = train_config.get('epochs', 10)
        batch_size = train_config.get('batch_size', 4)
        grad_accum_steps = train_config.get('grad_accum_steps', 4)
        lr = train_config.get('lr', 1e-4)
        start_epoch = self.configuration.get('start_epoch', 0)

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        resume_checkpoint = ''
        if start_epoch > 0:
            last_list = glob.glob(f"{data_path}/**/checkpoin.pth", recursive=True)
            resume_checkpoint = max(last_list, key=os.path.getctime) if last_list else ''
            logger.info(f'use checkpoint: {resume_checkpoint}')

        logger.info(
            f'Training configuration: epochs={epochs}, batch_size={batch_size}, lr={lr} , device_name={device_name}'
        )

        # Print directory structure of data_path up to 3 levels
        debug_msg = [f"-HHH- Printing directory structure of {data_path} (up to 3 levels):"]
        debug_msg.append(f"-HHH- Directory structure of {data_path}:")
        for root, dirs, files in os.walk(data_path, topdown=True):
            level = root.replace(data_path, '').count(os.sep)
            if level <= 3:
                indent = '  ' * level
                debug_msg.append(f"{indent}{os.path.basename(root)}/")
                if files:
                    subindent = '  ' * (level + 1)
                    for f in files:
                        debug_msg.append(f"{subindent}{f}")

        debug_output = '\n'.join(debug_msg)
        logger.info(debug_output)
        print(debug_output)

        # Flush stdout to ensure all logs are captured
        sys.stdout.flush()
        faas_callback = kwargs.get('on_epoch_end_callback')

        self.model.callbacks["on_fit_epoch_end"].append(lambda data: self.on_epoch_end(data, faas_callback))
        self.model.train(
            dataset_dir=data_path,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            resume=resume_checkpoint,
            output_dir=output_path,
            device=device_name,
            num_workers=0,
            # train crashed on : RuntimeError: Current CUDA Device does not support bfloat16. Please switch dtype to float16.
            # TODO check if that is best way to handle this
            fp16_eval=False,
            amp=False,
            dtype=torch.float16,
        )

        # Print directory structure of data_path up to 3 levels
        debug_msg = [f"-HHH- Printing directory structure of {output_path} (up to 3 levels):"]
        debug_msg.append(f"-HHH- Directory structure of {output_path}:")
        for root, dirs, files in os.walk(output_path, topdown=True):
            level = root.replace(output_path, '').count(os.sep)
            if level <= 3:
                indent = '  ' * level
                debug_msg.append(f"{indent}{os.path.basename(root)}/")
                if files:
                    subindent = '  ' * (level + 1)
                    for f in files:
                        debug_msg.append(f"{subindent}{f}")

        debug_output = '\n'.join(debug_msg)
        logger.info(debug_output)
        print(debug_output)

        #  Check if the model (checkpoint) has already completed training for the specified number of epochs, if so, can start again without resuming
        if 'start_epoch' in self.configuration and self.configuration['start_epoch'] == epochs:
            self.model_entity.configuration['start_epoch'] = 0
            self.model_entity.update()

        logger.info('Training completed')


if __name__ == '__main__':
    # Smart login with token handling
    # if dl.token_expired():
    #     dl.login()

    dl.login_api_key(
        api_key='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJlbWFpbCI6Imh1c2FtLm1AZGF0YWxvb3AuYWkiLCJpc3MiOiJodHRwczovL2dhdGUuZGF0YWxvb3AuYWkvMSIsImF1ZCI6Imh0dHBzOi8vZ2F0ZS5kYXRhbG9vcC5haS9hcGkvdjEiLCJpYXQiOjE3NDQwMzE5MjksImV4cCI6MTc3NDc5MDMyOSwic3ViIjoiYXBpa2V5fDM1YWQ0NWVjLTg2MjEtNGYxOC1iODc0LTJkMTFkZjdlZmI2MiIsImh0dHBzOi8vZGF0YWxvb3AuYWkvYXV0aG9yaXphdGlvbiI6eyJ1c2VyX3R5cGUiOiJhcGlrZXkiLCJyb2xlcyI6W119fQ.GlmZ1z9pjnDsdPoHc81inCZVJ-ZmZiwBS4gXfJIl3Ns2EwKl3LcJvDxUCU5ag6s_UpBhx1cSPJZhYe5eXrOjVORD1UJ4wPcVsd1_rzK5PN_skIsOjBdb7IngbAWWfW8cth_ByrKBWtEkGTwt40eN5FpCt-Wy7QP0spuBl_Tye7k3ReynSsO8au6W7qUm4PsvU4UHKWjywRQSH3usfOrsIwFJW4NyIAGdOyHS5Gekba1s26ZygOwlws5EeTFUAmWmLYKRYKh9K_n5e9uWHjgpbxkQsvnCAs9hnACudAMY37LN8KRgdmHOiaU-OZ1c7rSxx_S98a8hihXr2KYRbbm8zg'
    )

    project = dl.projects.get(project_name='ShadiDemo')
    # dataset = project.datasets.get(dataset_name='nvidia-husam-clone-updated-name')

    # model_id = '67fe3466f41fe3efebd2c433'
    # print('-HHH- get model')
    # model = project.models.get(model_id=model_id)
    # print('-HHH- create model adapter')
    # model_adapter = ModelAdapter(model)
    # predict_res = model_adapter.predict_items(
    #     items=[
    #         dataset.items.get(item_id='67ff9d8a18076275e55bd5ea'),
    #         dataset.items.get(item_id='67fbfb21489a0f6f359ee478'),
    #     ]
    # )

    # predict_res = model_adapter.predict_items(items=[dataset.items.get(item_id='67ff9d8a18076275e55bd5ea')])
    # print(f'-HHH- predict res: {predict_res}')
    model_name = 'rf-detr-clone-2604'
    model = project.models.get(model_name=model_name)
    model.status = 'pre-trained'
    model.update()
    print(f'-HHH- create model adapter')
    model_adapter = ModelAdapter(project.models.get(model_name=model_name))
    print(f'-HHH- run train model')
    model_adapter.train_model(model=model)
    print(f'-HHH- train model completed')

    # model_path = r'C:\Users\1Husam\.dataloop\models\rf-detr-tex4l\model.pth'

    # print("-HHH- 249")
    # model = RFDETRBase(pretrain_weights=model_path, device='cpu')
    # print("-HHH- 251")

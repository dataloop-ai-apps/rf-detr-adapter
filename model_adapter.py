import json
import sys
import logging
import os
import shutil
from typing import List, Any
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

    def save(self, local_path: str, **kwargs) -> None:
        self.configuration.update({'weights_filename': 'weights/best.pth'})

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

            self._process_coco_json(output_annotations_path)

            src_images_path = os.path.join(data_path, subset_name, 'items', subset_name)
            dst_images_path = os.path.join(data_path, dist_dir_name)
            self._copy_files(src_images_path, dst_images_path)

    def train(self, data_path: str, output_path: str, **kwargs) -> None:
        logger.info(f'Starting training with data from {data_path}')

        logger.info(f'-HHH- ver 28-apr data_path: {data_path}')
        print(f"-HHH- ver 28-apr data_path: {data_path}")
        epochs = self.configuration.get('epochs', 10)
        batch_size = self.configuration.get('batch_size', 4)
        grad_accum_steps = self.configuration.get('grad_accum_steps', 4)
        lr = self.configuration.get('lr', 1e-4)
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(
            f'Training configuration: epochs={epochs}, batch_size={batch_size}, lr={lr} , device_name={device_name}'
        )
        print("-HHH_ class_names", self.model_entity.labels)
        ##################### remove this ############################
        print("-HHH- 209 set batch size and grad accum steps to 1 for cpu")
        if device_name == 'cpu':
            batch_size = 1
            grad_accum_steps = 1
        ##################### remove this ############################
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

        def on_epoch_end(data):

            pass

        self.model.callbacks["on_fit_epoch_end"].append(on_epoch_end)
        self.model.train(
            dataset_dir=data_path,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=output_path,
            device=device_name,
            num_workers=0,
        )

        # if device_name == 'cpu':
        #     self.model.train(
        #         dataset_dir=data_path,
        #         epochs=epochs,
        #         batch_size=batch_size,
        #         grad_accum_steps=grad_accum_steps,
        #         lr=lr,
        #         output_dir=output_path,
        #         device=device_name,
        #         class_names=self.model_entity.labels,
        #         fp16_eval=False,
        #         amp=False,
        #         dtype=torch.float16,  # Use float16 for training
        #     )
        # else:
        #     self.model.train(
        #         dataset_dir=data_path,
        #         epochs=epochs,
        #         batch_size=batch_size,
        #         grad_accum_steps=grad_accum_steps,
        #         lr=lr,
        #         output_dir=output_path,
        #         device=device_name,
        #         class_names=self.model_entity.labels,
        #     )
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

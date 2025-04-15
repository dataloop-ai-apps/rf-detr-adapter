import logging
import os
import torch
import numpy as np
import requests
import dtlpy as dl

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

logger = logging.getLogger('rf-dert-adapter')


class ModelAdapter(dl.BaseModelAdapter):

    @staticmethod
    def _download_weights(url):
        if url is None:
            return None
        try:
            logger.info(f'Downloading weights from: {url}')
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Create temp file with .pth extension in weights dir
            model_filepath = os.path.join('/tmp/app/weights', f'model_{np.random.randint(0, 1000000)}.pth')

            with open(model_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f'Weights downloaded to: {model_filepath}')
            return model_filepath

        except (requests.exceptions.RequestException, IOError) as e:
            logger.error(f"Error downloading weights from {url}: {str(e)}")
            return None

    def load(self, local_path, **kwargs):
        """Load your model from saved weights"""
        model_filename = self.configuration.get('weights_filename', 'model.pth')
        model_filepath = os.path.normpath(os.path.join(local_path, model_filename))
        if not os.path.isfile(model_filepath):
            tmp_dir = '/tmp/app/weights'
            if os.path.isfile(tmp_dir + model_filename):
                model_filepath = tmp_dir + model_filename
            else:
                url = self.configuration.get('weights_url')
                # Download file to temporary location
                model_filepath = ModelAdapter._download_weights(url)

        self.confidence_threshold = self.configuration.get('conf_thres', 0.25)

        # not sure if self.device is needed
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(
            f'Confidence threshold: {self.confidence_threshold}, model_filepath: {model_filepath}, device: {device_name}'
        )
        self.model = RFDETRBase(pretrain_weights=model_filepath, device=device_name)

    # rf-dert is resize, normalize and convert to tensor in the model
    # nothing to do here
    # def prepare_item_func(self, item):
    #     pass

    def predict(self, batch, **kwargs):
        """Run predictions on a batch of data"""
        logger.info(f'Predicting batch of size: {len(batch)}')
        image_annotations = dl.AnnotationCollection()
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


if __name__ == '__main__':
    # Smart login with token handling
    # if dl.token_expired():
    #     dl.login()

    dl.login_api_key(
        api_key='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJlbWFpbCI6Imh1c2FtLm1AZGF0YWxvb3AuYWkiLCJpc3MiOiJodHRwczovL2dhdGUuZGF0YWxvb3AuYWkvMSIsImF1ZCI6Imh0dHBzOi8vZ2F0ZS5kYXRhbG9vcC5haS9hcGkvdjEiLCJpYXQiOjE3NDQwMzE5MjksImV4cCI6MTc3NDc5MDMyOSwic3ViIjoiYXBpa2V5fDM1YWQ0NWVjLTg2MjEtNGYxOC1iODc0LTJkMTFkZjdlZmI2MiIsImh0dHBzOi8vZGF0YWxvb3AuYWkvYXV0aG9yaXphdGlvbiI6eyJ1c2VyX3R5cGUiOiJhcGlrZXkiLCJyb2xlcyI6W119fQ.GlmZ1z9pjnDsdPoHc81inCZVJ-ZmZiwBS4gXfJIl3Ns2EwKl3LcJvDxUCU5ag6s_UpBhx1cSPJZhYe5eXrOjVORD1UJ4wPcVsd1_rzK5PN_skIsOjBdb7IngbAWWfW8cth_ByrKBWtEkGTwt40eN5FpCt-Wy7QP0spuBl_Tye7k3ReynSsO8au6W7qUm4PsvU4UHKWjywRQSH3usfOrsIwFJW4NyIAGdOyHS5Gekba1s26ZygOwlws5EeTFUAmWmLYKRYKh9K_n5e9uWHjgpbxkQsvnCAs9hnACudAMY37LN8KRgdmHOiaU-OZ1c7rSxx_S98a8hihXr2KYRbbm8zg'
    )

    project = dl.projects.get(project_name='ShadiDemo')
    dataset = project.datasets.get(dataset_name='nvidia-husam-clone-updated-name')

    model_id = '67fe3466f41fe3efebd2c433'
    print('-HHH- get model')
    model = project.models.get(model_id=model_id)
    print('-HHH- create model adapter')
    model_adapter = ModelAdapter(model)
    predict_res = model_adapter.predict_items(
        items=[
            dataset.items.get(item_id='67ff9d8a18076275e55bd5ea'),
            dataset.items.get(item_id='67fbfb21489a0f6f359ee478'),
        ]
    )

    predict_res = model_adapter.predict_items(items=[dataset.items.get(item_id='67ff9d8a18076275e55bd5ea')])
    print(f'-HHH- predict res: {predict_res}')

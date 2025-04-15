import dtlpy as dl
import torch
import os
import numpy as np

class SimpleModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load your model from saved weights"""
        print('🔄 Loading model from:', local_path)
        self.model = torch.load(os.path.join(local_path, 'model.pth'),weights_only=False)
        self.model.eval()
        print(f'-HHH- load 11 : {type(self.model)}')

    # TODO : no need to do this preprocess, rether need to override prepare item function prepare_item_func
    # which convert item to image.

    def preprocess(self, batch):
        # Convert batch to PyTorch tensor
        if isinstance(batch, list):
            # If batch is a list of numpy arrays
            batch = torch.tensor(batch, dtype=torch.float32)
        elif isinstance(batch, np.ndarray):
            # If batch is a single numpy array
            batch = torch.from_numpy(batch).float()
        
        # Reshape from NHWC to NCHW format
        if len(batch.shape) == 4:  # If batch is [N, H, W, C]
            batch = batch.permute(0, 3, 1, 2)  # Change to [N, C, H, W]
        
        return batch

    def predict(self, batch, **kwargs):
        """Run predictions on a batch of data"""
        print(f'🎯 Predicting batch of size: {len(batch)}')

        batch_tensor = self.preprocess(batch)
        
        # Get model predictions
        preds = self.model(batch_tensor)
        batch_predictions = torch.nn.functional.softmax(preds, dim=1)
        # Convert predictions to Dataloop format
        batch_annotations = list()
        for img_prediction in batch_predictions:
            pred_score, high_pred_index = torch.max(img_prediction, 0)
            pred_label = self.model_entity.id_to_label_map.get(int(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'dataset_id': self.model_entity.dataset_id})
            batch_annotations.append(collection)
            
        return batch_annotations
    
# if __name__ == '__main__':
#     from dotenv import load_dotenv

#     print('-HHH- load dotenv')
#     # Load environment variables from .env file
#     load_dotenv()

#     # Create a default configuration
#     dl.login_api_key(api_key=os.environ['DTLPY_API_KEY'])

#     project = dl.projects.get(project_name='ShadiDemo')
#     dataset = project.datasets.get(dataset_name='nvidia-husam-clone-updated-name')

#     model_id = '67fcc4cb06f7dc614feaf3e3'
#     print('-HHH- get model')
#     model = project.models.get(model_id=model_id)
#     print(f'-HHH- model: {model}')
#     print('-HHH- create model adapter')
#     model_adapter = SimpleModelAdapter(model)
#     item = dataset.items.get(item_id='67fbfd5b80e326df43dd9c03')
#     print(f'-HHH- get item: {item}')
#     predict_res = model_adapter.predict_items(items=[item])
#     print(f'-HHH- predict res: {predict_res}')


from model_adapter import Adapter
import dtlpymetrics
import dtlpy as dl
import unittest
import json
import glob
import os


class TestModelAdapter(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment. This method is called before each test.
        """
        self.project_root = os.getcwd()
        self.assets_path = os.path.join(self.project_root, "tests", "assets", "unittests")

    def prepare_item(self, local_item_name):
        matching_files = glob.glob(
            os.path.join(self.assets_path, f"{local_item_name}.*"))

        if len(matching_files) < 2:
            raise FileNotFoundError(
                f"Expected two files (JSON and media) for item '{local_item_name}', but found {len(matching_files)}"
            )

        # Identify JSON and other file
        manifest = None
        media_file = None

        for file_path in matching_files:
            if file_path.endswith(".json"):
                manifest = file_path
            else:
                media_file = file_path

        with open(manifest) as f:
            _json = json.load(f)

        item = dl.Item.from_json(
            _json=_json,
            client_api=dl.client_api,
        )

        return media_file, item  # Ultralytics can handle file path for prediction

    def test_inference_image(self, mean_score_threshold=0.9, precision_threshold=0.9, recall_threshold=0.9):
        local_item_name = 'image_item'
        model_path = os.path.join(self.project_root, "models", "yolo11", "dataloop.json")

        # Load model manifest
        with open(model_path) as f:
            manifest = json.load(f)
        models_json = manifest['components']['models']

        # Load Models from manifest
        for model_json in models_json:
            dummy_model = dl.Model.from_json(
                _json=model_json,
                client_api=dl.client_api,
                project=None,
                package=dl.Package()
            )

            os.chdir(self.project_root)
            adapter = Adapter(model_entity=dummy_model)
            item_stream, item = self.prepare_item(local_item_name=local_item_name)

            # Handling Annotations of dummy item
            output_collection = dl.AnnotationCollection(item=item)
            output_annotations = adapter.predict([(item_stream, item)])[0]
            for annotation in output_annotations:
                output_collection.add(annotation)

            # Predicted Annotations
            for ann in output_collection:
                ann.annotation_definition._item = item

            # Expected Annotations
            expected_collection = dl.AnnotationCollection.from_json_file(
                filepath=os.path.join(self.assets_path, f"{local_item_name}.json"))
            gt_collection = dl.AnnotationCollection(item=item)

            for ann in expected_collection:
                if ann.type == dummy_model.output_type:
                    gt_collection.add(ann)
            for ann in gt_collection:
                ann.annotation_definition._item = item

            # Compare annotations
            final_results = dtlpymetrics.utils.measure_annotations(annotations_set_one=gt_collection,
                                                                   annotations_set_two=output_collection)

            # Assertions for final results
            # Average similarity or match scores between predicted and ground-truth annotations.
            self.assertTrue(final_results.get("total_mean_score") >= mean_score_threshold,
                            msg=f"Total mean score is below the threshold: {final_results.get('total_mean_score')} < {mean_score_threshold}")

            # The proportion of ground truth annotations that are correctly matched by predictions
            self.assertTrue(final_results.get("precision") >= precision_threshold,
                            msg=f"Precision is below the threshold: {final_results.get('precision')} < {precision_threshold}")

            # The proportion of predicted annotations that correctly match ground truth.
            self.assertTrue(final_results.get("recall") >= recall_threshold,
                            msg=f"Recall is below the threshold: {final_results.get('recall')} < {recall_threshold}")

    def test_inference_video(self, mean_score_threshold=0.5, precision_threshold=0.5, recall_threshold=0.5):
        local_item_name = 'video_item'
        model_path = os.path.join(self.project_root, "models", "yolo11", "dataloop.json")

        # Load model manifest
        with open(model_path) as f:
            manifest = json.load(f)
        models_json = manifest['components']['models']

        # Load Models from manifest
        for model_json in models_json:
            dummy_model = dl.Model.from_json(
                _json=model_json,
                client_api=dl.client_api,
                project=None,
                package=dl.Package()
            )

            os.chdir(self.project_root)
            adapter = Adapter(model_entity=dummy_model)
            item_stream, item = self.prepare_item(local_item_name=local_item_name)

            # Handling Annotations of dummy item
            output_collection = dl.AnnotationCollection(item=item)
            output_annotations = adapter.predict([(item_stream, item)])[0]
            for annotation in output_annotations:
                output_collection.add(annotation)

            # Predicted Annotations
            for ann in output_collection:
                ann.annotation_definition._item = item

            # Expected Annotations
            expected_collection = dl.AnnotationCollection.from_json_file(
                filepath=os.path.join(self.assets_path, f"{local_item_name}.json"))
            gt_collection = dl.AnnotationCollection(item=item)

            for ann in expected_collection:
                if ann.type == dummy_model.output_type:
                    gt_collection.add(ann)
            for ann in gt_collection:
                ann.annotation_definition._item = item

            # Compare annotations
            final_results = dtlpymetrics.utils.measure_annotations(annotations_set_one=gt_collection,
                                                                   annotations_set_two=output_collection)

            # Assertions for final results
            # Average similarity or match scores between predicted and ground-truth annotations.
            self.assertTrue(final_results.get("total_mean_score") >= mean_score_threshold,
                            msg=f"Total mean score is below the threshold: {final_results.get('total_mean_score')} < {mean_score_threshold}")

            # The proportion of ground truth annotations that are correctly matched by predictions
            self.assertTrue(final_results.get("precision") >= precision_threshold,
                            msg=f"Precision is below the threshold: {final_results.get('precision')} < {precision_threshold}")

            # The proportion of predicted annotations that correctly match ground truth.
            self.assertTrue(final_results.get("recall") >= recall_threshold,
                            msg=f"Recall is below the threshold: {final_results.get('recall')} < {recall_threshold}")


if __name__ == '__main__':
    unittest.main()

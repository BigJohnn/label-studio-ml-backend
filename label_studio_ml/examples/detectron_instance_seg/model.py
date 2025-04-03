from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from PIL import Image
import label_studio_sdk
import torch, detectron2
import numpy as np
from detectron2.engine import DefaultTrainer

from detectron2.utils.logger import setup_logger
setup_logger()

import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import logging
logger = logging.getLogger(__name__)

class InstanceSegmentationModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

        # 初始化配置
        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL', 'http://192.168.100.159:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'a3f7c8e8cac209f797f9c65bc98c82c9d00a07c6')
        
        # 从标注配置解析标签
        self.labels = self.get_ordered_labels()
        logger.info(f'Loaded labels from config: {self.labels}')
        
    def get_ordered_labels(self) -> List[str]:
        """改进的标签解析方法"""
        from_name, to_name, value, labels = get_single_tag_keys(
            self.parsed_label_config, 
            control_tag='BrushLabels',
            object_tag='Image'
        )
        return labels
    
    def polygon_to_rle(self, points: List[float], width: int, height: int) -> List[int]:
        """将多边形坐标转换为RLE格式"""
        mask = np.zeros((height, width), dtype=np.uint8)
        contours = np.array(points).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [contours], 1)
        return self.rle_encode(mask)
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        path = self.get_local_path(tasks[0]['data']['image'], task_id=tasks[0]['id'])

        im = Image.open(path)
        outputs = predictor(im)
        results = []

        instances = outputs["instances"].to("cpu")
        # label_id = ?
        
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        width,height = im.size
        results.append({
            'id': label_id,
            'from_name': from_name,
            'to_name': to_name,
            'original_width': width,
            'original_height': height,
            'image_rotation': 0,
            'value': {
                'format': 'rle',
                'rle': rle,
                'brushlabels': [selected_label],
            },
            'type': 'brushlabels',
            'readonly': False
        })
            
        return [{
            'result': results,
            'model_version': self.get('model_version')
            # 'score': total_prob / max(len(results), 1)
        }]
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        if event == 'START_TRAINING':
            logger.info("Fitting model")
            ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
            project = ls.get_project(id=self.project_id)
            tasks = project.get_labeled_tasks()

            logger.info(f"Downloaded {len(tasks)} labeled tasks from Label Studio")

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("tangram_train",)
            cfg.DATASETS.TEST = ()
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
            cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
            cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
            cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
            cfg.SOLVER.STEPS = []        # do not decay learning rate
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
            # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(cfg) 
            trainer.resume_or_load(resume=False)
            trainer.train()
        print('fit() completed successfully.')


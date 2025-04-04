from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image
import label_studio_sdk
import torch, detectron2
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()

import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask as mask_utils  # 关键导入语句

from label_studio_ml.utils import get_single_tag_keys
import base64
import logging
logger = logging.getLogger(__name__)

def get_tangram_dicts(data_dir):
    dataset_dicts = []
    category_map = {"quad": 0,
                    "triangle_white":1,
                    "triangle_yellow":2,
                    "triangle_red":3,
                    "triangle_blue":4,
                    "triangle_green":5,
                    "parallelogram":6,
                    } 
    
    for task in InstanceSegmentationModel.processed_data:
        annotation = task['annotations'][0]
        if not annotation.get('result') or annotation.get('skipped') or annotation.get('was_cancelled'):
            continue

        # 提取基础图像信息
        x = annotation['result'][0]
        record = {
            "file_name": task['data']['image'],
            "image_id": task['id'],
            "height": x['original_height'],
            "width": x['original_width'],
            "annotations": []
        }

        # 遍历所有标注结果
        for res in annotation['result']:
            # 解码RLE并生成mask
            # mask = brush.decode_rle(res['value']['rle'])
            # mask = np.reshape(mask, [res['original_height'], res['original_width'], 4])[:, :, 3]
            mask = brush.decode_rle(res['value']['rle'])

            mask = mask.astype(np.uint8)
            # 步骤1：将一维数组转换为四通道形状 (H, W, 4)
            mask_4d = mask.reshape(res['original_height'], res['original_width'], 4)

            # 步骤2：提取Alpha通道（假设第4通道为有效掩膜）
            mask = mask_4d[:, :, 3]  # 索引从0开始，第4通道对应索引3

            # 步骤3：验证是否为二值化掩膜（0/255）
            mask = (mask > 0).astype(np.uint8) * 255

            # 生成COCO格式的RLE
            coco_rle = mask_utils.encode(np.asarray(mask, order="F", dtype=np.uint8))

            # 计算边界框（需要OpenCV）
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea)
            x_min, y_min, w, h = cv2.boundingRect(max_contour)

            x_max = x_min + w
            y_max = y_min + h
            ann = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_map[res['value']['brushlabels'][0]],
                "segmentation": coco_rle,
                "iscrowd": 0
            }
            record["annotations"].append(ann)

        dataset_dicts.append(record)
    return dataset_dicts
class InstanceSegmentationModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def __init__(self, project_id: Optional[str] = None, label_config=None):
        super(InstanceSegmentationModel, self).__init__(project_id, label_config)  # 自动解析配置
        print(self.parsed_label_config)  # 直接访问解析结果

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

        # 初始化配置
        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL', 'http://192.168.100.159:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'a3f7c8e8cac209f797f9c65bc98c82c9d00a07c6')
        print(f"Label Studio host: {self.LABEL_STUDIO_HOST}")
        print(f"Label Studio API key: {self.LABEL_STUDIO_API_KEY}")
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=self.project_id)
        tasks = project.get_labeled_tasks()

        logger.info(f"Downloaded {len(tasks)} labeled tasks from Label Studio")

        # images=[]
        # masks=[]
        # for task in tasks:
        #     for annotation in task['annotations']:
        #         if not annotation.get('result') or annotation.get('skipped') or annotation.get('was_cancelled'):
        #             continue

        #         path = self.get_local_path(task['data']['image'], task_id=task['id'])
        #         image = Image.open(path)
        #         images.append(image)

        #         res = task['annotations'][0]['result'][0]

        #         # image.save('detectron_instance_seg/images/image'+str(task['id'])+'.jpg')

        #         width = res['original_width']
        #         height = res['original_height']
        #         rle = res['value']['rle']

        #         # 示例RLE数据
        #         # shape = (width, height)  # 图像形状

        #         # 解码RLE
        #         mask = brush.decode_rle(rle)
        #         mask = np.reshape(mask, [height, width , 4])[:, :, 3]
        #         masks.append(Image.fromarray(mask/255))

                # # 创建PIL图像
                # pil_image = Image.fromarray(mask, mode='L')
                # # 保存为JPEG
                # pil_image.save('detectron_instance_seg/masks/mask'+str(task['id'])+'.jpg')

        # 将处理后的数据保存为类变量
        InstanceSegmentationModel.processed_data = tasks

        for d in ["train", "val"]:
            dataset_name = f"tangram_{d}"
            # 移除已存在的注册
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
            # TODO: tasks 的数据如何传给DatasetCatalog.register
            DatasetCatalog.register(dataset_name, 
                lambda d=d: get_tangram_dicts(os.path.join("tangram", d)))
            MetadataCatalog.get("tangram_" +d).set(thing_classes=["quad", 
                                                                    "triangle_white",
                                                                    "triangle_yellow",
                                                                    "triangle_red",
                                                                    "triangle_blue",
                                                                    "triangle_green",
                                                                    "parallelogram",
                                                                    ])
            
        InstanceSegmentationModel.tangram_metadata = MetadataCatalog.get("tangram_train")
        
    def polygon_to_rle(self, points: List[float], width: int, height: int) -> List[int]:
        """将多边形坐标转换为RLE格式"""
        mask = np.zeros((height, width), dtype=np.uint8)
        contours = np.array(points).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [contours], 1)
        return self.rle_encode(mask)
    
    # def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
    #     """ Write your inference logic here
    #         :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
    #         :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
    #         :return model_response
    #             ModelResponse(predictions=predictions) with
    #             predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
    #     """
    #     print(f'''\
    #     Run prediction on {tasks}
    #     Received context: {context}
    #     Project ID: {self.project_id}
    #     Label config: {self.label_config}
    #     Parsed JSON Label config: {self.parsed_label_config}
    #     Extra params: {self.extra_params}''')

    #     cfg = get_cfg()
    #     #  模型文件路径 ./output/model_final.pth 
    #     # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    #     # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #     predictor = DefaultPredictor(cfg)

    #     path = self.get_local_path(tasks[0]['data']['image'], task_id=tasks[0]['id'])

    #     im = Image.open(path)
    #     outputs = predictor(im)
    #     results = []

    #     instances = outputs["instances"].to("cpu")
    #     # label_id = ?
        
    #     from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
    #     width,height = im.size
    #     # results.append({
    #     #     'id': label_id,
    #     #     'from_name': from_name,
    #     #     'to_name': to_name,
    #     #     'original_width': width,
    #     #     'original_height': height,
    #     #     'image_rotation': 0,
    #     #     'value': {
    #     #         'format': 'rle',
    #     #         'rle': rle,
    #     #         'brushlabels': [selected_label],
    #     #     },
    #     #     'type': 'brushlabels',
    #     #     'readonly': False
    #     # })
            
    #     return [{
    #         'result': results,
    #         'model_version': self.get('model_version')
    #         # 'score': total_prob / max(len(results), 1)
    #     }]
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # 使用训练好的模型权重
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        predictor = DefaultPredictor(cfg)
        
        results = []
        category_map = {
            0: "quad",
            1: "triangle_white",
            2: "triangle_yellow",
            3: "triangle_red",
            4: "triangle_blue",
            5: "triangle_green",
            6: "parallelogram"
        }

        for task in tasks:
            # 获取图像路径和尺寸
            image_path = self.get_local_path(task['data']['image'], task_id=task['id'])
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # 执行预测
            outputs = predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # 解析模型输出
            pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
            scores = instances.scores.numpy() if instances.has("scores") else None
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
            pred_masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

            # self.parsed_label_config == {'tag2': {'type': 'KeyPointLabels', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['quad', 'triangle_white', 'triangle_yellow', 'triangle_red', 'triangle_blue', 'triangle_green', 'parallelogram'], 'labels_attrs': {'quad': {'value': 'quad', 'background': '#FF6B6B'}, 'triangle_white': {'value': 'triangle_white', 'background': '#FFFFFF'}, 'triangle_yellow': {'value': 'triangle_yellow', 'background': '#FFD700'}, 'triangle_red': {'value': 'triangle_red', 'background': '#FF0000'}, 'triangle_blue': {'value': 'triangle_blue', 'background': '#1890FF'}, 'triangle_green': {'value': 'triangle_green', 'background': '#52C41A'}, 'parallelogram': {'value': 'parallelogram', 'background': '#B37FEB'}}}, 'tag3': {'type': 'RectangleLabels', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['quad', 'triangle_white', 'triangle_yellow', 'triangle_red', 'triangle_blue', 'triangle_green', 'parallelogram'], 'labels_attrs': {'quad': {'value': 'quad', 'background': '#FF6B6B'}, 'triangle_white': {'value': 'triangle_white', 'background': '#FFFFFF'}, 'triangle_yellow': {'value': 'triangle_yellow', 'background': '#FFD700'}, 'triangle_red': {'value': 'triangle_red', 'background': '#FF0000'}, 'triangle_blue': {'value': 'triangle_blue', 'background': '#1890FF'}, 'triangle_green': {'value': 'triangle_green', 'background': '#52C41A'}, 'parallelogram': {'value': 'parallelogram', 'background': '#B37FEB'}}}, 'tagx': {'type': 'PolygonLabels', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['quad', 'triangle_white', 'triangle_yellow', 'triangle_red', 'triangle_blue', 'triangle_green', 'parallelogram'], 'labels_attrs': {'quad': {'value': 'quad', 'background': '#FF6B6B'}, 'triangle_white': {'value': 'triangle_white', 'background': '#FFFFFF'}, 'triangle_yellow': {'value': 'triangle_yellow', 'background': '#FFD700'}, 'triangle_red': {'value': 'triangle_red', 'background': '#FF0000'}, 'triangle_blue': {'value': 'triangle_blue', 'background': '#1890FF'}, 'triangle_green': {'value': 'triangle_green', 'background': '#52C41A'}, 'parallelogram': {'value': 'parallelogram', 'background': '#B37FEB'}}}, 'tag': {'type': 'BrushLabels', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['quad', 'triangle_white', 'triangle_yellow', 'triangle_red', 'triangle_blue', 'triangle_green', 'parallelogram'], 'labels_attrs': {'quad': {'value': 'quad', 'background': '#FF6B6B'}, 'triangle_white': {'value': 'triangle_white', 'background': '#FFFFFF'}, 'triangle_yellow': {'value': 'triangle_yellow', 'background': '#FFD700'}, 'triangle_red': {'value': 'triangle_red', 'background': '#FF0000'}, 'triangle_blue': {'value': 'triangle_blue', 'background': '#1890FF'}, 'triangle_green': {'value': 'triangle_green', 'background': '#52C41A'}, 'parallelogram': {'value': 'parallelogram', 'background': '#B37FEB'}}}}
            brush_config = {
                k: v for k, v in self.parsed_label_config.items() 
                if v['type'] == 'BrushLabels'
            }
            assert len(brush_config) == 1, "必须存在且仅存在一个BrushLabels标签"

            # 使用过滤后的配置调用函数
            from_name, to_name, value, labels = get_single_tag_keys(
                parsed_label_config=brush_config,
                control_type="BrushLabels", 
                object_type="Image"
            )
            print(f"from_name: {from_name}, to_name: {to_name}, value: {value}, labels: {labels}")

            task_results = []
            for i in range(len(instances)):
                if scores[i] < 0.5:  # 过滤低置信度结果
                    continue

                # 转换掩膜为RLE格式[1](@ref)
                mask = pred_masks[i].astype(np.uint8) * 255

                # Save the mask as an image for debugging purposes
                mask_image_path = f"mask_{task['id']}_{i}.png"
                cv2.imwrite(mask_image_path, mask)
                print(f"Saved mask image to {mask_image_path}")

                rle = mask_utils.encode(np.asfortranarray(mask))  # 确保内存连续
                # rle['counts'] = rle['counts'].decode('utf-8')  # 转换为字符串格式
                rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
                print(f"RLE: {rle}") #rle是乱码， 如何转成字符串数字
                #TODO....................................................................................
                
                # 获取类别标签
                class_id = pred_classes[i]
                label = category_map.get(class_id, "unknown")

                # 构建结果字典[1,4](@ref)
                task_results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'brushlabels',
                    'value': {
                        'format': 'rle',
                        'rle': rle,
                        'brushlabels': [label],
                        'width': width,
                        'height': height
                    },
                    'score': float(scores[i])
                })

            results.append({
                'result': task_results,
                'model_version': self.get('model_version'),
                'score': float(np.mean(scores)) if scores.any() else 0.0
            })
        
        return ModelResponse(predictions=results)
    
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
            cfg.INPUT.MASK_FORMAT = "bitmask"
            # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(cfg) 
            trainer.resume_or_load(resume=False)
            trainer.train()
        print('fit() completed successfully.')


import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(os.path.abspath('projects/GroundedSAM2'))
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from grounding_dino.groundingdino.util.inference import load_model
import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import re



class VideoObjectTracker:
    def __init__(self, expression):
        from transformers.utils import logging
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = 'sam2_hiera_l.yaml'
        self.sam2_checkpoint = 'projects/GroundedSAM2/checkpoints/sam2_hiera_large.pt'
        self.grounding_dino_model_id = 'IDEA-Research/grounding-dino-base'

        self.output_dir_path = 'videos/outputs'
        self.output_video_path = 'videos/outputs/output.mp4'
        self.step = 15
        self.text = expression + '.'
        self.name_to_label = self._create_name_to_label_dict()

        # Initialize models
        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        self.sam2_image_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(self.device)
        
        # Create necessary directories
        self.mask_data_dir = os.path.join(self.output_dir_path, "mask_data")
        self.json_data_dir = os.path.join(self.output_dir_path, "json_data")
        self.result_dir = os.path.join(self.output_dir_path, "result")
        self.key_image_with_box_data_dir = os.path.join(self.output_dir_path, "key_image_with_box")
        self.output_bbox_json_path = os.path.join(self.output_dir_path, "bbox.json")
        CommonUtils.creat_dirs(self.mask_data_dir)
        CommonUtils.creat_dirs(self.json_data_dir)
        CommonUtils.creat_dirs(self.result_dir)
        CommonUtils.creat_dirs(self.key_image_with_box_data_dir)
        

    def _create_name_to_label_dict(self):
        labels = [label.strip() for label in self.text.split(".") if label.strip()]
        name_to_label = {label: idx for idx, label in enumerate(labels)}

        return name_to_label

    def _generate_instance_bbox_json(self):
        instance_dict = {}
        image_info = {}

        mask_pattern = re.compile(r'mask_(\d+)')

        file_names = sorted([f for f in os.listdir(self.json_data_dir) if f.endswith(".json")])
        file_length = len(file_names)

        for filename in file_names:
            file_path = os.path.join(self.json_data_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                mask_height = data.get('mask_height')
                mask_width = data.get('mask_width')
                if mask_height and mask_width: 
                    image_info['mask_height'] = mask_height
                    image_info['mask_width'] = mask_width

                match = mask_pattern.search(filename)
                if match:
                    frame_number = match.group(1)
                    frame_idx = int(frame_number)
                    new_frame_name = f"{frame_number}.jpg"

                for instance_id, instance_info in data.get('labels', {}).items():
                    bbox = {
                        "frame_name": new_frame_name,
                        "bbox": [instance_info['x1'], instance_info['y1'], instance_info['x2'], instance_info['y2']]
                    }
                    if instance_id not in instance_dict:
                        instance_dict[instance_id] = {
                            "category": instance_info['class_name'],
                            "pred_labels": self.name_to_label[instance_info['class_name']],
                            "pred_scores": instance_info.get('logit', 0.0),
                            "bboxes": [{} for _ in range(file_length)]
                        }

                    instance_dict[instance_id]["bboxes"][frame_idx] = bbox

        sorted_instance_dict = {}
        for instance_id, data in instance_dict.items():
            sorted_instance_dict[instance_id] = {
                "category": data["category"],
                "pred_labels": data["pred_labels"],
                "pred_scores": data["pred_scores"],
                "bboxes": data["bboxes"]
            }
        
        result = {
            'image_info': image_info,
            'instances': sorted_instance_dict
        }
        
        with open(self.output_bbox_json_path, 'w') as output_file:
            json.dump(result, output_file, indent=4)



    def _generate_outputs_from_json(self):
        file_path = self.output_bbox_json_path
        print(file_path)
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            image_info = data.get('image_info', {})
            instances = data.get('instances', {})
        
            image_height = image_info.get('mask_height', 0)
            image_width = image_info.get('mask_width', 0)
            image_size = (image_height, image_width)

            pred_scores = []
            pred_labels = []
            pred_bboxes = []

            for instance_id, instance_info in instances.items():
                pred_score = instance_info.get('pred_scores', 0.0)
                pred_scores.append(pred_score)

                pred_label = instance_info.get('pred_labels', -1)
                pred_labels.append(pred_label)

                pred_bbox = instance_info.get('bboxes', [])
                bboxes_each_frame = []
                for bbox in pred_bbox:
                    bbox_each_frame = bbox.get('bbox', None)
                    if bbox_each_frame is not None:
                        bbox_each_frame = torch.tensor(bbox_each_frame)
                    bboxes_each_frame.append(bbox_each_frame)   
                pred_bboxes.append(bboxes_each_frame)
            
            result = {
                'image_size': image_size,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
                'pred_bboxes': pred_bboxes,
                'frame_captions': [],  # 如果有captions数据，可在此填充
                # 'pred_masks': []  # 如果有masks数据，可在此填充
            }
            
            return result

    def __call__(self, frames_dir):
        """
        Tracks objects in the given video using SAM and Grounding DINO models.
        """

        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        last_masks_per_step = MaskDictionaryModel()
        objects_count = 0

        frame_names = [
            p for p in os.listdir(frames_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = self.video_predictor.init_state(video_path=frames_dir)
        PROMPT_TYPE_FOR_VIDEO = "mask"

        start_frame_idx = 0
        while start_frame_idx < len(frame_names):
            img_path = os.path.join(frames_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame_idx].split(".")[0]

            key_masks_per_step = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

            # Run Grounding DINO on the image
            inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # Prompt SAM image predictor to get the mask for the object
            self.image_predictor.set_image(np.array(image.convert("RGB")))

            input_boxes = results[0]["boxes"]
            OBJECTS = results[0]["labels"]

            # Optional: Visualize Grounding DINO detection boxes
            draw = ImageDraw.Draw(image)
            for box in input_boxes:
                box = box.cpu().numpy().astype(int)
                draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)

            output_image_path = os.path.join(self.key_image_with_box_data_dir, f"{image_base_name}_with_boxes.jpg")
            image.save(output_image_path)
            print(f"Saved image with bounding boxes to {output_image_path}")

            # 如果没有检测到物体，保存文件并跳过当前帧
            if len(input_boxes) == 0:
                print(f"No object detected in the frame, moving to next frame {start_frame_idx + 1}")
                
                mask_img = np.zeros(image.size[::-1], dtype=np.uint16)
                np.save(os.path.join(self.mask_data_dir, f"mask_{image_base_name}.npy"), mask_img)

                json_data = {"labels": {}}
                json_data_path = os.path.join(self.json_data_dir, f"mask_{image_base_name}.json")
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

                start_frame_idx += 1
                continue
            
            # 如果检测到了物体，继续执行后续的 mask 生成和保存逻辑
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            if key_masks_per_step.promote_type == "mask":
                key_masks_per_step.add_new_frame_annotation(mask_list=torch.tensor(masks).to(self.device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only supports mask prompts")


            objects_count = key_masks_per_step.update_masks(tracking_annotation_dict=last_masks_per_step, iou_threshold=0.8, objects_count=objects_count)
            
            # 这个函数似乎没必要
            if len(key_masks_per_step.labels) == 0:
                print(f"No object detected in the frame, skipping frame {start_frame_idx}")
                continue

            self.video_predictor.reset_state(inference_state)

            for object_id, object_info in key_masks_per_step.labels.items():
                _, _, _ = self.video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )

            video_segments = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=self.step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=key_masks_per_step.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                last_masks_per_step = copy.deepcopy(frame_masks)

            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(self.mask_data_dir, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(self.json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

            start_frame_idx += self.step

        # CommonUtils.draw_masks_and_box_with_supervision(frames_dir, self.mask_data_dir, self.json_data_dir, self.result_dir)
        # create_video_from_images(self.result_dir, self.output_video_path, frame_rate=30)
        self._generate_instance_bbox_json()
        tracker_outputs = self._generate_outputs_from_json()

        return self.output_bbox_json_path, tracker_outputs

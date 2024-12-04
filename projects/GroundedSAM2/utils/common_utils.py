import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random

class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    # 对子文件夹的处理可以优化
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        
        # Function to recursively get all image files
        def get_all_images(directory):
            all_images = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):  # Adjust file extensions as necessary
                        all_images.append(os.path.join(root, file))
            return all_images

        # Get all image paths recursively
        raw_image_paths = get_all_images(raw_image_path)
        raw_image_paths.sort()

        for image_path in raw_image_paths:

            raw_image_name = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy")
            mask = np.load(mask_npy_path)
            
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0:  # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            if len(all_object_masks) == 0:
                print(f"No object detected in the frame, the frame {raw_image_name} does not have annotation.")
                annotated_frame = image.copy()
            else:
                all_object_masks = np.concatenate(all_object_masks, axis=0)
            
                # load box information
                file_path = os.path.join(json_path, "mask_" + raw_image_name.split(".")[0] + ".json")
                
                all_object_boxes = []
                all_object_ids = []
                all_class_names = []
                object_id_to_name = {}
                with open(file_path, "r") as file:
                    json_data = json.load(file)
                    for obj_id, obj_item in json_data["labels"].items():
                        # box id
                        instance_id = obj_item["instance_id"]
                        if instance_id not in unique_ids:  # not a valid box
                            continue
                        # box coordinates
                        x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                        all_object_boxes.append([x1, y1, x2, y2])
                        # box name
                        class_name = obj_item["class_name"]
                        
                        # build id list and id2name mapping
                        all_object_ids.append(instance_id)
                        all_class_names.append(class_name)
                        object_id_to_name[instance_id] = class_name
                
                # Adjust object id and boxes to ascending order
                paired_id_and_box = zip(all_object_ids, all_object_boxes)
                sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
                
                # Because we get the mask data as ascending order, so we also need to ascend box and ids
                all_object_ids = [pair[0] for pair in sorted_pair]
                all_object_boxes = [pair[1] for pair in sorted_pair]
                
                detections = sv.Detections(
                    xyxy=np.array(all_object_boxes),
                    mask=all_object_masks,
                    class_id=np.array(all_object_ids, dtype=np.int32),
                )
                
                # custom label to show both id and class name
                labels = [
                    f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
                ]
                
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

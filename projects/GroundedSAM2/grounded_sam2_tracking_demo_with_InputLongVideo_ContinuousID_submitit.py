import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm


# TODO: 解决grounding-DINO错误框定问题，如果没检测到物体怎么办；解决prompt不一致的报错问题
# TODO: 对物体遮挡再重现的一致性做的不好

def track_objects_in_video(args):

    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # save video frames to JPEG images
    total_clip_nums = save_video_frames_to_jpg_with_clip(args.mp4_dir, args.video_dir, args.clip_frames, stride=1, start=0, end=None)
    print("total clip numbers:", total_clip_nums)

    # create the output directory
    CommonUtils.creat_dirs(args.output_dir_path)
    mask_data_dir = os.path.join(args.output_dir_path, "mask_data")
    json_data_dir = os.path.join(args.output_dir_path, "json_data")
    result_dir = os.path.join(args.output_dir_path, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    # Optional: 可视化 Grounding DINO 检测框
    key_image_with_box_data_dir = os.path.join(args.output_dir_path, "key_image_with_box")
    CommonUtils.creat_dirs(key_image_with_box_data_dir)


    last_masks_per_clip = MaskDictionaryModel()
    objects_count = 0
    # 对每个clip进行处理
    for dir_clip in range(total_clip_nums):

        """
        # init model
        """

        video_predictor = build_sam2_video_predictor(args.model_cfg, args.sam2_checkpoint)
        sam2_image_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        # init grounding dino model from huggingface
        processor = AutoProcessor.from_pretrained(args.grounding_dino_model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_dino_model_id).to(device)

        video_clip_dir = os.path.join(args.video_dir, f"clip_{dir_clip:04d}")

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_clip_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        # 可以不需要
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # init video predictor state
        inference_state = video_predictor.init_state(video_path=video_clip_dir)

        PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point

        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
        """
        print("frames per clip:", len(frame_names))

        last_masks_per_step = MaskDictionaryModel()

        # 因为可能key帧检测不到物体，而SAM又必须要prompt，所以检测不到时，key帧变成下一帧
        # for start_frame_idx in range(0, len(frame_names), step):
        start_frame_idx = 0
        while start_frame_idx < len(frame_names):
        # prompt grounding dino to get the box coordinates on specific frame

            # continue
            img_path = os.path.join(video_clip_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame_idx].split(".")[0]
            # 每一step关键帧的masks
            key_masks_per_step = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
            # run Grounding DINO on the image
            inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            input_boxes = results[0]["boxes"] # .cpu().numpy()
            # print("results[0]",results[0])
            OBJECTS = results[0]["labels"]

            # Optional：可视化 Grounding DINO 检测框
            draw = ImageDraw.Draw(image)
            for box in input_boxes:
                box = box.cpu().numpy().astype(int)  # 转换为numpy并转换为整型
                draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)

            # 保存结果图像
            output_image_path = os.path.join(key_image_with_box_data_dir, f"{image_base_name}_with_boxes.jpg")
            image.save(output_image_path)
            print(f"Saved image with bounding boxes to {output_image_path}")


            # 如果没有检测到物体，保存文件并跳过当前帧
            if len(input_boxes) == 0:
                print("No object detected in the frame, moving to next frame {}".format(start_frame_idx + 1))
                
                # 保存空白的 mask 和 JSON 文件
                mask_img = np.zeros(image.size[::-1], dtype=np.uint16)
                np.save(os.path.join(mask_data_dir, f"mask_{image_base_name}.npy"), mask_img)

                json_data = {"labels": {}}
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)
                
                # 增加帧索引
                start_frame_idx += 1
                continue

            # 如果检测到了物体，继续执行后续的 mask 生成和保存逻辑

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            # 3 还是 2 ？？？？？？？？？？？？？？
            # 3 报错 masks shape: torch.Size([1, 360, 480]) ； mask_img shape: torch.Size([360, 480])
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor
            """
            # If you are using point prompts, we uniformly sample positive points based on the mask
            if key_masks_per_step.promote_type == "mask":
                key_masks_per_step.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")

            print(dir_clip, "start_frame_idx", start_frame_idx)
            # 新的clip开始
            if start_frame_idx == 0:
                # 很神奇，设置一样的threshold，有的物体就检测不出来
                objects_count = key_masks_per_step.update_masks(tracking_annotation_dict=last_masks_per_clip, iou_threshold=0.2, objects_count=objects_count)
            else:
            # 新的step开始
                objects_count = key_masks_per_step.update_masks(tracking_annotation_dict=last_masks_per_step, iou_threshold=0.8, objects_count=objects_count)
            print("objects_count", objects_count)

            if len(key_masks_per_step.labels) == 0:
                print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                
                continue

            video_predictor.reset_state(inference_state)

            for object_id, object_info in key_masks_per_step.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
                
            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """

            
            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=args.step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = key_masks_per_step.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                # 优化，不用每次都copy
                last_masks_per_step = copy.deepcopy(frame_masks)
                print('clip', dir_clip, 'step', args.step, "out_frame_idx", out_frame_idx)
                # 有一个问题在于step最后一个step有的帧数可能不等于step
                # 例如step=15，最后一个step可能只有5帧
                # if out_frame_idx == start_frame_idx + step:
                #     last_masks_per_step = copy.deepcopy(frame_masks)

            
            """
            Step 5: save the tracking masks and json files
            """
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

            start_frame_idx += args.step

        print("clip", dir_clip, "done")
        last_masks_per_clip = copy.deepcopy(last_masks_per_step)
        del image, masks, logits, inputs, input_boxes, results, video_segments, img_path, image_base_name, key_masks_per_step, OBJECTS, grounding_model, processor, inference_state
        del video_predictor, sam2_image_model, image_predictor, frame_names, video_clip_dir
        torch.cuda.empty_cache()

    """
    Step 6: Draw the results and save the video
    """
    CommonUtils.draw_masks_and_box_with_supervision(args.video_dir, mask_data_dir, json_data_dir, result_dir)

    create_video_from_images(result_dir, args.output_video_path, frame_rate=30)


def save_video_frames_to_jpg_with_clip(mp4_dir, video_dir, clip_frames=1000, stride=1, start=0, end=None):
    # Generate frames from the video
    frame_generator = sv.get_video_frames_generator(mp4_dir, stride=stride, start=start, end=end)
    
    # Ensure the base directory exists
    base_dir = Path(video_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Initialize counters for subdirectories and images
    sub_dir_index = 0
    image_counter = 0
    sub_dir = base_dir / f"clip_{sub_dir_index:04d}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Save video frames as JPEG images
    with sv.ImageSink(
        target_dir_path=sub_dir, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            # Save the current frame
            sink.save_image(frame)
            image_counter += 1
            
            # Check if we need to switch to a new subdirectory
            if image_counter >= clip_frames:
                sub_dir_index += 1
                image_counter = 0
                sub_dir = base_dir / f"clip_{sub_dir_index:04d}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                sink.target_dir_path = sub_dir  # Update sink to new subdirectory

    # Scan and sort all JPEG frame names in the directory
    frame_names = []
    for sub_dir in base_dir.iterdir():
        if sub_dir.is_dir():
            sub_frame_names = [
                str(sub_dir / p) for p in os.listdir(sub_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ]
            frame_names.extend(sorted(sub_frame_names, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])))

    clip_nums = sub_dir_index + 1

    return clip_nums


def save_video_frames_to_jpg(mp4_dir, video_dir, stride=1, start=0, end=None):
    # Generate frames from the video
    frame_generator = sv.get_video_frames_generator(mp4_dir, stride=stride, start=start, end=end)
    
    # Create directory for saving frames
    source_frames = Path(video_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    # Save video frames as JPEG images
    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    # Scan and sort all JPEG frame names in the directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return len(frame_names)


def get_parser():
    parser = argparse.ArgumentParser(description='Chat-Video', add_help=False)
    parser.add_argument('--config', type=str, default='configs/grounded_sam2_tracking_demo.yaml', help='Path to config file')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt', help='Path to SAM 2 checkpoint')
    parser.add_argument('--model_cfg', type=str, default='sam2_hiera_l.yaml', help='Model configuration')
    parser.add_argument('--grounding_dino_model_id', type=str, default="IDEA-Research/grounding-dino-tiny", help='Grounding DINO model ID(IDEA-Research/grounding-dino-tiny)')
    parser.add_argument('--video_dir', type=str, default='tracking_frames', help='Directory of video frames which will be created with filenames like `<frame_index>.jpg')
    parser.add_argument('--output_dir_path', type=str, default='outputs', help='Directory to save output')
    parser.add_argument('--output_video_path', type=str, default='outputs/output.mp4', help='Path to save final video')
    parser.add_argument('--mp4_dir', type=str, default='assets/G-HRgYT6CCY.mp4', help='Path to input MP4 video')
    parser.add_argument('--clip_frames', type=int, default=600, help='Number of frames to ONE clip')
    parser.add_argument('--step', type=int, default=15, help='Step size for KEY FRAME sampling')
    parser.add_argument('--text', type=str, default='people.', help='Text prompt for SAM 2 and Grounding DINO')
    
    return parser
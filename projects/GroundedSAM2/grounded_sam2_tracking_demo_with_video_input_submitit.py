import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse

def track_objects_in_video(args):
    """
    Tracks objects in a video using SAM2 and Grounding DINO models.

    Parameters:
    - video_dir: Directory containing video frames as JPEG images.
    - text: Text prompt for object detection.
    - sam2_checkpoint: Path to SAM2 model checkpoint.
    - model_cfg: Configuration file for SAM2 model.
    - grounding_model_id: Model ID for Grounding DINO.
    - save_dir: Directory to save annotated frames.
    - output_video_path: Path to save the output video.
    """

    # Use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize SAM2 video and image predictor models
    video_predictor = build_sam2_video_predictor(args.model_cfg, args.sam2_checkpoint)
    sam2_image_model = build_sam2(args.model_cfg, args.sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # Initialize Grounding DINO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model_id).to(device)

    # # Scan all the JPEG frame names in the directory
    # # 直接视频为图片帧的
    # frame_names = [p for p in os.listdir(args.video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    """
    Custom video input directly using video files
    """
    frame_generator = sv.get_video_frames_generator(args.video_dir, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(args.source_video_frame_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(args.source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize video predictor state
    inference_state = video_predictor.init_state(video_path=args.source_video_frame_dir)

    ann_frame_idx = 0  # The frame index to interact with
    ann_obj_id = 1  # Give a unique ID to each object (can be any integer)

    # Get the box and mask for a specific frame
    img_path = os.path.join(args.source_video_frame_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # Run Grounding DINO on the image
    inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    # Register each object's positive points to the video predictor
    PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
        # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
    

    # Propagate the video predictor to get the segmentation results for each frame
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Visualize and save the segmentation results across the video
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(args.source_video_frame_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks), # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(args.save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # Convert the annotated frames to video
    create_video_from_images(args.save_dir, args.output_video_path)


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training', add_help=False)
    parser.add_argument('--video_dir', default='./assets/G-HRgYT6CCY.mp4', type=str, help='directory containing video frames as JPEG images')
    parser.add_argument('--text', default='man.', type=str, help='text prompt for object detection')
    parser.add_argument('--sam2_checkpoint', default='./checkpoints/sam2_hiera_large.pt', type=str, help='path to SAM2 model checkpoint')
    parser.add_argument('--model_cfg', default='sam2_hiera_l.yaml', type=str, help='configuration file for SAM2 model')
    parser.add_argument('--grounding_model_id', default='IDEA-Research/grounding-dino-tiny', type=str, help='model ID for Grounding DINO')
    parser.add_argument('--save_dir', default='./tracking_results', type=str, help='directory to save annotated frames')
    parser.add_argument('--output_video_path', default='./AA_tracking_demo_video.mp4', type=str, help='path to save the output video')
    parser.add_argument('--source_video_frame_dir', default="./custom_video_frames1", type=str, help='directory containing video frames as JPEG images')
    return parser
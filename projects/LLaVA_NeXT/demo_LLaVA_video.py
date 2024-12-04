import copy
from projects.GroundedSAM2.sam2.modeling import sam
from regex import F
import torch
import numpy as np
from PIL import Image
import warnings
from decord import VideoReader, cpu
import cv2
import sys
sys.path.append('projects/LLaVA_NeXT/')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

class LLaVAVideoDescriber:
    def __init__(self, max_frames=16, pretrained_path="lmms-lab/llava-onevision-qwen2-0.5b-ov", model_name="llava_qwen", device="cuda", device_map="auto"):
        self.pretrained_path = pretrained_path
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        self.max_frames = max_frames
        
        # Load model
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.pretrained_path, None, self.model_name, device_map=self.device_map, attn_implementation="sdpa"
        )
        self.model.eval()

    def load_video(self, video_path, max_frames_num):
        if isinstance(video_path, str):
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            raise ValueError("Invalid video path.")
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def load_frames(self, frames, max_frames_num):
        if type(frames) != list:
            raise ValueError("Invalid frames. It should be a list of frames.")
        total_frame_num = len(frames)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        sampled_frames = frames[frame_idx]
        return sampled_frames  # (frames, height, width, channels)


    def generate(self, video, question, type='video_path', conv_template="qwen_1_5"):
        if type == 'video_path':
            video_frames = self.load_video(video, self.max_frames)
        elif type == 'video_frames':
            video_frames = self.load_frames(video, self.max_frames)
        elif type == 'video_array':
            video_frames = video
        else:
            raise ValueError("Invalid type. Choose from 'video' and 'frames'.")
        print(video_frames.shape) # (frames, height, width, channels)
        image_tensors = []
        frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(self.device)
        image_tensors.append(frames)

        processed_question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], processed_question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [(frame.shape[2], frame.shape[1]) for frame in image_tensors]  # (width, height)
        modalities = ["video"] * len(image_tensors)

        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=modalities,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]
    
def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    success, frame = cap.read()
    
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = cap.read()
    
    cap.release()
    frames_array = np.array(frames)
    
    return frames_array


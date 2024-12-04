import copy
import torch
from PIL import Image
import warnings
import sys
sys.path.append('projects/LLaVA_NeXT/')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

class LLaVADescriber:
    def __init__(self, pretrained_path="lmms-lab/llava-onevision-qwen2-0.5b-ov", model_name="llava_qwen", device="cuda", device_map="auto"):
        self.pretrained_path = pretrained_path
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.pretrained_path, None, self.model_name, device_map=self.device_map
        )
        self.model.eval()

    def process_image(self, image):
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        return image_tensor, image.size

    def generate(self, image, question, conv_template="qwen_1_5"):
        image_tensor, image_size = self.process_image(image)

        question = f"{DEFAULT_IMAGE_TOKEN} \n{question}"
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]


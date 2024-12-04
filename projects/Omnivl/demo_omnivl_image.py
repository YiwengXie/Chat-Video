import os
import math
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from projects.Omnivl.models.blip_vqa import blip_vqa
from projects.BLIP2.demo_blip2_caption import select_device


class OmniVLVQAPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, config, device=None):
        if device is None:
            selected_device = select_device()
        else:
            selected_device = device

        device = torch.device(selected_device)
        print("Creating OmniVL VQA model on device: ", selected_device)
        model = blip_vqa(
            config=config,
            pretrained=config["pretrained"],
            image_size=config["image_size"],
            num_frames=config.get("input_length", 1),
            temporal_stride=config.get("temporal_stride", 1),
            vit_grad_ckpt=config["vit_grad_ckpt"],
            vit_ckpt_layer=config["vit_ckpt_layer"],
            enable_mae=config["enable_mae"],
        )
        model = model.to(device)
        self.omnivl_model = model

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["image_size"], config["image_size"]),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.device = device

    def ask(self, image, question):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.omnivl_model(
                image,
                question,
                None,
                train=False,
                inference="generate",
            )
            answer = outputs[0]

        return answer

    def __call__(self, frames, question):
        caption_list = []
        for frm in frames:
            frm_caption = self.ask(frm, question)
            caption_list.append(frm_caption)

        return caption_list

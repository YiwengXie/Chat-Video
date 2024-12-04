"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import logging
import os
import warnings
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from projects.Omnivl.models.med import BertConfig, BertLMHeadModel, BertModel
from projects.Omnivl.models.timesformer import TimeSformer
from projects.Omnivl.models.vit import VisionTransformer, interpolate_pos_embed
from timm.models.hub import download_cached_file
from torch import nn
from transformers import BertTokenizer

warnings.filterwarnings("ignore")


class BLIP_Base(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=224,
        num_frames=1,
        temporal_stride=1,
        encoder_name="vit-base",
        cls_on=True,
        embed_dim=256,
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        enable_mae=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.cls_on = cls_on
        self.visual_encoder, vision_width = create_visual_encoder(
            encoder_name,
            image_size,
            num_frames,
            temporal_stride,
            vit_grad_ckpt,
            vit_ckpt_layer,
            enable_mae=enable_mae,
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(
            config=med_config, add_pooling_layer=False
        )

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

    def encode_text(self, caption, device):
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )

        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])

        return text_feat

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)

        if self.cls_on:
            image_feat = self.vision_proj(image_embeds[:, 0, :])
        else:
            image_feat = self.vision_proj(image_embeds.mean(dim=1))

        return image_feat

    def forward(self, image, caption, mode):

        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == "image":
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == "text":
            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            return text_output.last_hidden_state

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(image.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(
        self,
        med_config="projects/Omnivl/configs/med_config.json",
        image_size=384,
        encoder_name="vit-base",
        num_frames=1,
        temporal_stride=1,
        cls_on=True,
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt="a picture of ",
        enable_mae=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        # print(num_frames)
        self.visual_encoder, vision_width = create_visual_encoder(
            encoder_name,
            image_size,
            num_frames,
            temporal_stride,
            vit_grad_ckpt,
            vit_ckpt_layer,
            enable_mae=enable_mae,
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        ).to(image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )
        loss_lm = decoder_output.loss

        return loss_lm

    @autocast()  # for mixed precision
    def generate(
        self,
        image,
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        customized_prompt=None,
    ):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        if customized_prompt is not None:
            this_prompt = customized_prompt
        else:
            this_prompt = self.prompt

        prompt = [this_prompt] * image_embeds.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            image.device
        )
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs,
            )
        else:
            # beam search
            # print(input_ids.shape, model_kwargs["encoder_hidden_states"].shape)
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs,
            )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(this_prompt) :])
        return captions


def blip_decoder(config, pretrained="", **kwargs):
    model = BLIP_Decoder(
        encoder_name=config["encoder_name"], cls_on=config["cls_on"], **kwargs
    )
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, config)

    return model


def init_tokenizer():
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("projects/UNINEXT/bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_visual_encoder(
    encoder_name,
    image_size=224,
    num_frames=1,
    temporal_stride=1,
    use_grad_checkpointing=False,
    ckpt_layer=0,
    drop_path_rate=0,
    enable_mae=False,
    decoder_embed_dim=768,
    decoder_depth=1,
    decoder_num_heads=16,
):

    assert encoder_name in [
        "vit-base",
        "vit-large",
        "timesformer",
        "t2d",
        "meter",
        "fit",
        "davit-tiny",
        "davit-base",
        "davit-base-t2d",
    ]

    if encoder_name == "vit-base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

    elif encoder_name == "vit-large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

    elif encoder_name == "timesformer":
        vision_width = 768
        # print("timesformer: ", num_frames)
        visual_encoder = TimeSformer(
            img_size=image_size,
            patch_size=16,
            num_frames=num_frames,
            temporal_stride=temporal_stride,
            attention_type="divided_space_time",
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

    else:
        raise NotImplementedError

    return visual_encoder, vision_width


def blip(config, pretrained="", **kwargs):
    model = BLIP_Base(
        encoder_name=config["encoder_name"], cls_on=config["cls_on"], **kwargs
    )
    if pretrained:
        model, msg = load_checkpoint(model, pretrained, config)
    return model


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename, config):
    encoder_name = config["encoder_name"]
    task_type = config["task_type"]
    inflate = config.get("inflate", False)

    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")

    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    state_dict = (
        checkpoint["model"]
        if "model" in checkpoint.keys()
        else checkpoint["state_dict"]
    )

    if is_url(url_or_filename):
        new_state_dict = state_dict.copy()
        for k in list(new_state_dict):
            new_key = "visual_encoder." + k
            new_state_dict[new_key] = state_dict[k]

            new_key2 = "visual_encoder_m." + k
            new_state_dict[new_key2] = state_dict[k]

            del new_state_dict[k]

        state_dict = new_state_dict

    if config.get("inflate", False) or "retrieval" in task_type:
        if "queue_str" in state_dict.keys():
            del state_dict["queue_ptr"]
        if "queue_str2" in state_dict.keys():
            del state_dict["queue_ptr2"]

    if encoder_name in ["vit-base", "vit-large", "timesformer"]:
        if inflate:
            temporal_stride = model.visual_encoder.temporal_stride

            patch_embed2d = state_dict["visual_encoder.patch_embed.proj.weight"]
            patch_embed3d = (
                patch_embed2d.unsqueeze(2).repeat(1, 1, temporal_stride, 1, 1)
                / temporal_stride
            )
            state_dict[
                "visual_encoder.patch_embed_3d.proj.weight"
            ] = patch_embed3d
            state_dict["visual_encoder.patch_embed_3d.proj.bias"] = state_dict[
                "visual_encoder.patch_embed.proj.bias"
            ]

            patch_embed2d_m = state_dict[
                "visual_encoder_m.patch_embed.proj.weight"
            ]
            patch_embed3d_m = (
                patch_embed2d_m.unsqueeze(2).repeat(1, 1, temporal_stride, 1, 1)
                / temporal_stride
            )
            state_dict[
                "visual_encoder_m.patch_embed_3d.proj.weight"
            ] = patch_embed3d_m
            state_dict[
                "visual_encoder_m.patch_embed_3d.proj.bias"
            ] = state_dict["visual_encoder_m.patch_embed.proj.bias"]

            print(
                "inflate the patch embedding from {} to {}.".format(
                    1, temporal_stride
                )
            )

        if config["enable_mae"]:
            del state_dict["visual_encoder.pos_embed"]
            del state_dict["visual_encoder.decoder_pos_embed"]

            if "visual_encoder_m.pos_embed" in model.state_dict().keys():
                del state_dict["visual_encoder_m.pos_embed"]
                del state_dict["visual_encoder_m.decoder_pos_embed"]

        else:
            state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )

            if (
                "visual_encoder_m.pos_embed" in model.state_dict().keys()
                and "visual_encoder_m.pos_embed" in state_dict.keys()
            ):
                state_dict[
                    "visual_encoder_m.pos_embed"
                ] = interpolate_pos_embed(
                    state_dict["visual_encoder_m.pos_embed"],
                    model.visual_encoder_m,
                )

        if "visual_encoder.time_embed" in state_dict.keys():
            temporal_stride = model.visual_encoder.temporal_stride
            if (
                model.visual_encoder.num_frames // temporal_stride
            ) != state_dict["visual_encoder.time_embed"].size(1):
                print(
                    "resize the temporal pos embedding from {} to {}".format(
                        state_dict["visual_encoder.time_embed"].size(1),
                        model.visual_encoder.num_frames // temporal_stride,
                    )
                )
                time_embed = state_dict["visual_encoder.time_embed"].transpose(
                    1, 2
                )
                new_time_embed = F.interpolate(
                    time_embed,
                    size=(model.visual_encoder.num_frames // temporal_stride),
                    mode="linear",
                )
                state_dict[
                    "visual_encoder.time_embed"
                ] = new_time_embed.transpose(1, 2)

            """elif "image" in task_type:
                del state_dict["visual_encoder.time_embed"]"""

        if "visual_encoder_m.time_embed" in state_dict.keys():
            if hasattr(model, "visual_encoder_m"):
                temporal_stride = model.visual_encoder_m.temporal_stride
                if (
                    model.visual_encoder_m.num_frames // temporal_stride
                ) != state_dict["visual_encoder_m.time_embed"].size(1):
                    time_embed = state_dict[
                        "visual_encoder_m.time_embed"
                    ].transpose(1, 2)
                    new_time_embed = F.interpolate(
                        time_embed,
                        size=(
                            model.visual_encoder_m.num_frames // temporal_stride
                        ),
                        mode="linear",
                    )
                    state_dict[
                        "visual_encoder_m.time_embed"
                    ] = new_time_embed.transpose(1, 2)

                """elif "image" in task_type:
                    del state_dict["visual_encoder_m.time_embed"]"""

        if inflate and "video" in task_type:
            if model.visual_encoder.attention_type == "divided_space_time":
                new_state_dict = state_dict.copy()
                for key in list(new_state_dict):
                    if "blocks" in key and "attn" in key:
                        new_key = key.replace("attn", "temporal_attn")
                        if not new_key in state_dict:
                            new_state_dict[new_key] = state_dict[key]
                        else:
                            new_state_dict[new_key] = state_dict[new_key]

                    if "blocks" in key and "norm1" in key:
                        new_key = key.replace("norm1", "temporal_norm1")
                        if not new_key in state_dict:
                            new_state_dict[new_key] = state_dict[key]
                        else:
                            new_state_dict[new_key] = state_dict[new_key]

                state_dict = new_state_dict

        elif "video" in task_type:
            new_state_dict = state_dict.copy()
            for key in state_dict:
                if key.endswith("image_queue"):
                    del new_state_dict[key]

                elif key.endswith("text_queue"):
                    del new_state_dict[key]

                elif "image_queue2" in key:
                    new_key = key.replace("image_queue2", "image_queue")
                    new_state_dict[new_key] = state_dict[key]
                    del new_state_dict[key]

                elif "text_queue2" in key:
                    new_key = key.replace("text_queue2", "text_queue")
                    new_state_dict[new_key] = state_dict[key]
                    del new_state_dict[key]

            state_dict = new_state_dict

    msg = model.load_state_dict(state_dict, strict=False)
    # print("load checkpoint from %s" % url_or_filename, msg)
    del checkpoint
    torch.cuda.empty_cache()

    return model, msg

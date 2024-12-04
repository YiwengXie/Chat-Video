"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import math
import os
import time
from typing import List

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from einops import rearrange
from projects.Omnivl.models.blip import create_visual_encoder, init_tokenizer
from projects.Omnivl.models.med import BertConfig, BertLMHeadModel, BertModel
from projects.Omnivl.models.utils import (
    concat_all_gather,
    patchify,
    smooth_l1_loss,
    tie_encoder_decoder_weights,
    unpatchify,
)
from projects.Omnivl.models.vit import Block
from timm.models.layers import trunc_normal_
from torch import nn
from transformers import BertTokenizer
from utils import is_main_process

transformers.logging.set_verbosity_error()


class BLIP_Pretrain(nn.Module):
    def __init__(
        self,
        med_config="configs/bert_config.json",
        cls_on=True,
        image_models=None,
        pretrained=True,
        share_weights=True,
        image_size=224,
        num_frames=1,
        temporal_stride=1,
        encoder_name="vit-base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        momentum=0.995,
        enable_mae=False,
        decoder_embed_dim=768,
        decoder_depth=1,
        decoder_num_heads=16,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.cls_on = cls_on
        if cls_on:
            print("Using CLS token for visual encoder.")
        else:
            print("No CLS token for visual encoder.")

        self.enable_mae = enable_mae
        self.visual_encoder, vision_width = create_visual_encoder(
            encoder_name,
            image_size,
            num_frames,
            temporal_stride,
            vit_grad_ckpt,
            vit_ckpt_layer,
            0,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.image_models = image_models

        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        # encoder_config.hidden_size = vision_width
        # encoder_config.num_attention_heads = vision_width // 64

        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        print(
            "Width of visual encoder: {}, width of text encoder: {}".format(
                vision_width, text_width
            )
        )

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_visual_encoder(
            encoder_name,
            image_size,
            num_frames,
            temporal_stride,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(
            config=encoder_config, add_pooling_layer=False
        )
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("image_queue2", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue2", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.image_queue2 = nn.functional.normalize(self.image_queue2, dim=0)
        self.text_queue2 = nn.functional.normalize(self.text_queue2, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=decoder_config
        )

        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        if share_weights:
            tie_encoder_decoder_weights(
                self.text_encoder, self.text_decoder.bert, "", "/attention"
            )

        else:
            print("Not sharing weights between text encoder and decoder")

        self.initialize_weights()

    def initialize_weights(self):
        if self.pretrained:
            if self.encoder_name == "vit-base":
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu",
                    check_hash=True,
                )
                state_dict = checkpoint["model"]
                msg = self.visual_encoder.load_state_dict(
                    state_dict, strict=False
                )
                print(msg)

            elif self.encoder_name == "vit-large":
                from timm.models.helpers import load_custom_pretrained
                from timm.models.vision_transformer import default_cfgs

                load_custom_pretrained(
                    self.visual_encoder,
                    default_cfgs["vit_large_patch16_224_in21k"],
                )

            elif self.encoder_name in ["davit-tiny", "davit-base"]:
                if self.image_models:
                    state_dict = torch.load(
                        self.image_models, map_location="cpu"
                    )
                    state_dict = (
                        state_dict["state_dict"]
                        if "state_dict" in state_dict.keys()
                        else state_dict
                    )
                    new_state_dict = state_dict.copy()
                    for k in list(new_state_dict):
                        if "patch_embeds" in k:
                            new_k = k.replace("patch_embeds", "conv_embeds")
                            new_state_dict[new_k] = state_dict[k]
                            del new_state_dict[k]

                    state_dict = new_state_dict

                    msg = self.visual_encoder.load_state_dict(
                        state_dict, strict=False
                    )
                    print(msg)

        else:
            print("Training visual encoder from scratch...")

    def forward(self, image, caption, alpha, target=None, mask_ratio=None):
        if len(image.shape) == 5:
            mode = "video"

        else:
            mode = "image"

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        bs = image.size(0)
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if self.cls_on:
            image_feat = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1
            )
        else:
            image_feat = F.normalize(
                self.vision_proj(image_embeds.mean(dim=1)), dim=-1
            )

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(image.device)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        # B, C
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)

            if self.cls_on:
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
                )
            else:
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m.mean(dim=1)), dim=-1
                )

            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]),
                dim=-1,
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        if mode == "image":
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        else:
            self._dequeue_and_enqueue2(image_feat_m, text_feat_m)

        loss_mae = None
        if self.enable_mae:
            assert mask_ratio is not None
            pred_x, mask = self.visual_encoder(image, mask_ratio)
            loss_mae = smooth_l1_loss(image_embeds_m, pred_x, mask)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long),
            ],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= LM ========================##
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_output = self.text_decoder(
            decoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        loss_lm = decoder_output.loss
        return loss_ita, loss_itm, loss_lm, loss_mae

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # print('deque for image feature...')
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, image_feat, text_feat):
        # print('deque for video feature...')
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue2[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue2[:, ptr : ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr2[0] = ptr


class BLIP_Pretrain_UniCL(nn.Module):
    def __init__(
        self,
        med_config="configs/bert_config.json",
        cls_on=True,
        image_models=None,
        pretrained=True,
        share_weights=True,
        image_size=224,
        num_frames=1,
        temporal_stride=1,
        encoder_name="vit-base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        momentum=0.995,
        enable_mae=False,
        decoder_embed_dim=768,
        decoder_depth=1,
        decoder_num_heads=16,
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
            0,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

        self.enable_mae = enable_mae

        self.pretrained = pretrained
        self.image_models = image_models

        self.encoder_name = encoder_name
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        # encoder_config.hidden_size = vision_width
        # encoder_config.num_attention_heads = vision_width // 64

        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        print(
            "Width of visual encoder: {}, width of text encoder: {}".format(
                vision_width, text_width
            )
        )

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_visual_encoder(
            encoder_name,
            image_size,
            num_frames,
            temporal_stride,
            enable_mae=enable_mae,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(
            config=encoder_config, add_pooling_layer=False
        )
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), 0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("image_queue2", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue2", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue2", torch.full((1, queue_size), 0))
        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.image_queue2 = nn.functional.normalize(self.image_queue2, dim=0)
        self.text_queue2 = nn.functional.normalize(self.text_queue2, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=decoder_config
        )
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        if share_weights:
            tie_encoder_decoder_weights(
                self.text_encoder, self.text_decoder.bert, "", "/attention"
            )

        else:
            print("Not sharing weights between text encoder and decoder")

        self.initialize_weights()

    def initialize_weights(self):
        if self.pretrained:
            if self.encoder_name == "vit-base":
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu",
                    check_hash=True,
                )
                state_dict = checkpoint["model"]
                msg = self.visual_encoder.load_state_dict(
                    state_dict, strict=False
                )
                print(msg)

            elif self.encoder_name == "vit-large":
                from timm.models.helpers import load_custom_pretrained
                from timm.models.vision_transformer import default_cfgs

                load_custom_pretrained(
                    self.visual_encoder,
                    default_cfgs["vit_large_patch16_224_in21k"],
                )

            elif self.encoder_name in ["davit-tiny", "davit-base"]:
                if self.image_models:
                    state_dict = torch.load(
                        self.image_models, map_location="cpu"
                    )
                    state_dict = (
                        state_dict["state_dict"]
                        if "state_dict" in state_dict.keys()
                        else state_dict
                    )
                    new_state_dict = state_dict.copy()
                    for k in list(new_state_dict):
                        if "patch_embeds" in k:
                            new_k = k.replace("patch_embeds", "conv_embeds")
                            new_state_dict[new_k] = state_dict[k]
                            del new_state_dict[k]

                    state_dict = new_state_dict

                    msg = self.visual_encoder.load_state_dict(
                        state_dict, strict=False
                    )
                    print(msg)

        else:
            print("Training visual encoder from scratch...")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image, caption, alpha, target=None, mask_ratio=None):
        if len(image.shape) == 5:
            mode = "video"

        else:
            mode = "image"

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if self.cls_on:
            image_feat = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1
            )
        else:
            image_feat = F.normalize(
                self.vision_proj(image_embeds.mean(dim=1)), dim=-1
            )

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(image.device)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)

            if self.cls_on:
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
                )
            else:
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m.mean(dim=1)), dim=-1
                )

            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]),
                dim=-1,
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            """sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)"""

            target = target.view(-1, 1)
            target_all = torch.cat(
                [target.t(), self.idx_queue.clone().detach()], dim=1
            )
            target_all = target_all[target_all >= 0]
            target_modified = target_all.clone().view(-1)
            supervised_data = target_all[target_all > 0]
            max_label = supervised_data.max() if supervised_data.numel() else 0

            target_modified[target_all == 0] = (
                max_label
                + torch.arange(0, (target_all == 0).sum()).type_as(target_all)
                + 1
            )

            idx = target_modified[: target.size(0)].view(-1, 1)
            idx_all = target_modified.view(1, -1)

            sim_targets = torch.eq(idx, idx_all).float()

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        if mode == "image":
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, target)

        else:
            self._dequeue_and_enqueue2(image_feat_m, text_feat_m, target)

        loss_mae = None
        if self.enable_mae:
            assert mask_ratio is not None

            pred_x, mask = self.visual_encoder(image, mask_ratio)
            loss_mae = smooth_l1_loss(image_embeds_m, pred_x, mask)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4

            target_ = target_modified[:bs]
            mask = target_.view(-1, 1) == target_.view(1, -1)
            weights_t2i = weights_t2i.masked_fill_(mask, 0)
            weights_i2t = weights_i2t.masked_fill_(mask, 0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long),
            ],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= LM ========================##
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_output = self.text_decoder(
            decoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        loss_lm = decoder_output.loss

        return loss_ita, loss_itm, loss_lm, loss_mae

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # print('deque for image feature...')
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idxs)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr : ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, image_feat, text_feat, idxs):
        # print('deque for video feature...')
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idxs)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue2[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue2[:, ptr : ptr + batch_size] = text_feats.T
        self.idx_queue2[:, ptr : ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr2[0] = ptr


def blip_pretrain(contrast_type, **kwargs):
    if contrast_type == "blip":
        model = BLIP_Pretrain(**kwargs)

    else:
        model = BLIP_Pretrain_UniCL(**kwargs)

    return model

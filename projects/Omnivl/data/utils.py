import io
import json
import math
import numbers
import os
import pickle
import random
import re
from enum import Enum

import numpy as np
from PIL import Image

import av
import projects.Omnivl.data.decoder as decoder
import decord
import mmcv
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import projects.Omnivl.utils as utils
from decord import VideoReader
# from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torch.nn.functional import interpolate as img_tensor_resize
from torch.nn.functional import pad as img_tensor_pad
from torch.nn.modules.utils import _quadruple
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


class VideoNorm(object):
    """Apply Normalization to Image Pixels on GPU"""

    def __init__(
        self,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ):
        # self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (N, 3, H, W)
        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.0)
        re = img.sub_(self.mean).div_(self.std)
        return re.permute(1, 0, 2, 3)


def get_video_decoding_kwargs(
    container,
    num_frames,
    target_fps,
    num_clips=None,
    clip_idx=None,
    sampling_strategy="rand",
    safeguard_duration=False,
    video_max_pts=None,
):
    if num_clips is None:
        three_clip_names = ["start", "middle", "end"]  # uniformly 3 clips
        assert sampling_strategy in ["rand", "uniform"] + three_clip_names
        if sampling_strategy == "rand":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=-1,  # random sampling
                num_clips=None,  # will not be used when clip_idx is `-1`
                target_fps=target_fps,
            )
        elif sampling_strategy == "uniform":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,  # will not be used when clip_idx is `-2`
                num_frames=num_frames,
                clip_idx=-2,  # uniformly sampling from the whole video
                num_clips=1,  # will not be used when clip_idx is `-2`
                target_fps=target_fps,  # will not be used when clip_idx is `-2`
            )
        else:  # in three_clip_names
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=three_clip_names.index(sampling_strategy),
                num_clips=3,
                target_fps=target_fps,
            )
    else:  # multi_clip_ensemble, num_clips and clip_idx are only used here
        assert clip_idx is not None
        # sampling_strategy will not be used, as uniform sampling will be used by default.
        # uniformly sample `num_clips` from the video,
        # each clip sample num_frames frames at target_fps.
        decoder_kwargs = dict(
            container=container,
            sampling_rate=1,
            num_frames=num_frames,
            clip_idx=clip_idx,
            num_clips=num_clips,
            target_fps=target_fps,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts,
        )
    return decoder_kwargs


def extract_frames_from_video_path(
    video_path,
    num_clips,
    clip_idx,
    target_fps=3,
    num_frames=3,
    multi_thread_decode=False,
    sampling_strategy="rand",
    safeguard_duration=False,
    transform=None,
):
    with open(video_path, "rb") as f:
        input_bytes = f.read()
    in_mem_bytes_io = io.BytesIO(input_bytes)
    frames, _ = extract_frames_from_video_binary(
        in_mem_bytes_io,
        target_fps=target_fps,
        num_frames=num_frames,
        num_clips=num_clips,
        clip_idx=clip_idx,
        multi_thread_decode=multi_thread_decode,
        sampling_strategy=sampling_strategy,
        safeguard_duration=safeguard_duration,
        transform=transform,
    )
    return frames


def extract_frames_from_video_binary(
    in_mem_bytes_io,
    target_fps=3,
    num_frames=3,
    num_clips=None,
    clip_idx=None,
    multi_thread_decode=False,
    sampling_strategy="rand",
    safeguard_duration=False,
    video_max_pts=None,
    transform=None,
):
    """
    Args:
        in_mem_bytes_io: binary from read file object
            >>> with open(video_path, "rb") as f:
            >>>     input_bytes = f.read()
            >>> frames = extract_frames_from_video_binary(input_bytes)
            OR from saved binary in lmdb database
            >>> env = lmdb.open("lmdb_dir", readonly=True)
            >>> txn = env.begin()
            >>> stream = io.BytesIO(txn.get(str("key").encode("utf-8")))
            >>> frames = extract_frames_from_video_binary(stream)
            >>> from torchvision.utils import save_image
            >>> save_image(frames[0], "path/to/example.jpg")  # save the extracted frames.
        target_fps: int, the input video may have different fps, convert it to
            the target video fps before frame sampling.
        num_frames: int, number of frames to sample.
        multi_thread_decode: bool, if True, perform multi-thread decoding.
        sampling_strategy: str, how to sample frame from video, one of
            ["rand", "uniform", "start", "middle", "end"]
            `rand`: randomly sample consecutive num_frames from the video at target_fps
                Note it randomly samples a clip containing num_frames at target_fps,
                not uniformly sample from the whole video
            `uniform`: uniformly sample num_frames of equal distance from the video, without
                considering target_fps/sampling_rate, etc. E.g., when sampling_strategy=uniform
                and num_frames=3, it samples 3 frames at [0, N/2-1, N-1] given a video w/ N frames.
                However, note that when num_frames=1, it will sample 1 frame at [0].
                Also note that `target_fps` will not be used under `uniform` sampling strategy.
            `start`/`middle`/`end`: first uniformly segment the video into 3 clips, then sample
                num_frames from the corresponding clip at target_fps. E.g., num_frames=3, a video
                w/ 30 frames, it samples [0, 1, 2]; [9, 10, 11]; [18, 19, 20] for start/middle/end.
            If the total #frames at target_fps in the video/clip is less than num_frames,
            there will be some duplicated frames
        num_clips: int,
        clip_idx: int
        safeguard_duration:
        video_max_pts: resue it to improve efficiency
    Returns:
        torch.uint8, (T, C, H, W)
    """
    try:
        # Add `metadata_errors="ignore"` to ignore metadata decoding error.
        # When verified visually, it does not seem to affect the extracted frames.
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    except Exception as e:
        print(f"Exception in loading video binary: {e}")
        return None, None

    if multi_thread_decode:
        # Enable multiple threads for decoding.
        video_container.streams.video[0].thread_type = "AUTO"
    # (T, H, W, C), channels are RGB
    # see docs in decoder.decode for usage of these parameters.
    decoder_kwargs = get_video_decoding_kwargs(
        container=video_container,
        num_frames=num_frames,
        target_fps=target_fps,
        num_clips=num_clips,
        clip_idx=clip_idx,
        sampling_strategy=sampling_strategy,
        safeguard_duration=safeguard_duration,
        video_max_pts=video_max_pts,
    )
    frames, video_max_pts = decoder.decode(**decoder_kwargs)

    frames_trans = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames_trans.append(frame)

    # T C H W -> C T H W
    video = torch.stack(frames_trans).permute(1, 0, 2, 3).contiguous()

    return video, video_max_pts


def load_video_from_path_decord(
    video_path,
    frm_sampling_strategy,
    num_frm,
    height=None,
    width=None,
    start_time=None,
    end_time=None,
    fps=-1,
    return_fps=False,
):
    try:
        if not height or not width:
            vr = VideoReader(video_path)
        else:
            vr = VideoReader(video_path, width=width, height=height)

        vlen = len(vr)

        if start_time or end_time:
            assert (
                fps > 0
            ), "must provide video fps if specifying start and end time."
            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)

        else:
            start_idx, end_idx = 0, vlen

        if frm_sampling_strategy == "uniform":
            frame_indices = np.arange(
                start_idx, end_idx, vlen / num_frm, dtype=int
            )
        elif frm_sampling_strategy == "nlvl_uniform":
            frame_indices = np.arange(
                start_idx, end_idx, vlen / num_frm
            ).astype(int)
        elif frm_sampling_strategy == "nlvl_rand":
            frame_indices = np.arange(
                start_idx, end_idx, vlen / num_frm
            ).astype(int)

            # generate some random perturbations
            strides = [
                frame_indices[i] - frame_indices[i - 1]
                for i in range(1, len(frame_indices))
            ] + [vlen - frame_indices[-1]]
            pertube = np.array(
                [np.random.randint(0, stride) for stride in strides]
            )

            frame_indices = frame_indices + pertube

        elif frm_sampling_strategy == "rand":
            frame_indices = sorted(random.sample(range(vlen), num_frm))
        elif frm_sampling_strategy == "headtail":
            frame_indices_head = sorted(
                random.sample(range(vlen // 2), num_frm // 2)
            )
            frame_indices_tail = sorted(
                random.sample(range(vlen // 2, vlen), num_frm // 2)
            )
            frame_indices = frame_indices_head + frame_indices_tail
        else:
            """raise NotImplementedError(
                "Invalid sampling strategy {} ".format(frm_sampling_strategy)
            )"""
            frame_indices = np.arange(start_idx, end_idx, 1, dtype=int)

        raw_sample_frms = vr.get_batch(frame_indices)

    except Exception as e:
        return None

    # T H W C
    if isinstance(raw_sample_frms, torch.Tensor):
        raw_sample_frms = raw_sample_frms.numpy().astype(np.uint8)
    else:
        raw_sample_frms = raw_sample_frms.asnumpy().astype(np.uint8)

    if return_fps:
        return raw_sample_frms, vr.get_avg_fps()

    else:
        return raw_sample_frms


def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.
    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim (bool): If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`. Default: ``True``
    Returns:
        torch.Tensor: tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError(
            "Input type must be a numpy.ndarray. Got {}".format(type(image))
        )

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array"
        )

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape)
        )

    return tensor.unsqueeze(0) if not keepdim else tensor


def get_padding(image, max_w, max_h, pad_all=False):
    # keep the images to upper-left corner
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_padding, v_padding = max_w - w, max_h - h
    if pad_all:
        h_padding /= 2
        v_padding /= 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    else:
        l_pad, t_pad = 0, 0
        r_pad, b_pad = h_padding, v_padding
    if isinstance(image, torch.Tensor):
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    else:
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class ImagePad(object):
    def __init__(self, max_w, max_h, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.max_w = max_w
        self.max_h = max_h
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        if isinstance(img, torch.Tensor):
            paddings = _quadruple(get_padding(img, self.max_w, self.max_h))
            return img_tensor_pad(img, paddings, self.padding_mode, self.fill)
        return img_pad(
            img,
            get_padding(img, self.max_w, self.max_h),
            self.fill,
            self.padding_mode,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )


def get_resize_size(image, max_size):
    """
    Args:
        image: PIL Image or torch.tensor
        max_size:

    Returns:

    Note the height/width order difference
    >>> pil_img = Image.open("raw_img_tensor.jpg")
    >>> pil_img.size
    (640, 480)  # (width, height)
    >>> np_img = np.array(pil_img)
    >>> np_img.shape
    (480, 640, 3)  # (height, width, 3)
    """
    # note the order of height and width for different inputs
    if isinstance(image, torch.Tensor):
        # width, height = image.shape[-2:]
        height, width = image.shape[-2:]
    else:
        width, height = image.size

    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))
    return size


class VideoRandomSquareCrop(object):
    def __init__(self, crop_size, p=0.5):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        self.p = p

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be cropped.

        Returns:
            torch.tensor: cropped video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                b, t, h, w = video.shape
            else:
                raise RuntimeError(
                    "Expecting 4-dimensional tensor of shape (b,t,h,w), got {}".format(
                        video.shape
                    )
                )

            # if random.uniform(0, 1) < self.p:
            #     video = torch.flip(video, (3,))

            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, :, x : x + self.crop_size, y : y + self.crop_size]

        else:
            raise NotImplementedError(
                "Support only torch.Tensor as input, got {}".format(type(video))
            )


class VideoResizeSquare(object):
    def __init__(self, out_size, interpolation="nearest"):
        assert isinstance(out_size, int)
        self.out_size = out_size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be scaled.

        Returns:
            torch.tensor: Rescaled video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                t, c, h, w = video.shape
                assert (
                    c == 3
                ), "Expecting 3-channel color video, got video of shape {}".format(
                    video.shape
                )
            else:
                raise RuntimeError(
                    "Expecting 4-dimensional tensor of shape (b,t,h,w), got {}".format(
                        video.shape
                    )
                )

            short_side = h if h < w else w
            # scaling_factor = self.out_size / short_side

            # new_h = int(h * scaling_factor)
            # new_w = int(w * scaling_factor)

            resized_video = img_tensor_resize(
                video,
                size=((self.out_size, self.out_size)),
                mode=self.interpolation,
            )

            return resized_video

        else:
            # in other data class, the order of shape might be different.
            raise NotImplementedError(
                "Support only torch.Tensor as input, got {}".format(type(video))
            )

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.out_size, self.interpolation
        )


class ImageResize(object):
    """Resize the input image (torch.tensor) to the given size.

    Args:
        max_size (int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_size, interpolation=Image.BILINEAR):
        assert isinstance(max_size, int)
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (torch.tensor): Image to be scaled.

        Returns:
            torch.tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            assert isinstance(self.interpolation, str)
            return img_tensor_resize(
                img,
                size=get_resize_size(img, self.max_size),
                mode=self.interpolation,
                align_corners=False,
            )
        return img_resize(
            img, get_resize_size(img, self.max_size), self.interpolation
        )

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.size, interpolate_str
        )


def get_imagenet_transform(min_size=600, max_size=1000):
    """parameters from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This simply crop the center square from the image
    """
    if min_size != 600:
        import warnings

        warnings.warn(
            f"Warning: min_size is not used in image transform, "
            f"setting min_size will have no effect."
        )
    return transforms.Compose(
        [
            ImageResize(
                max_size, Image.BILINEAR
            ),  # longer side will be resized to 1000
            ImagePad(max_size, max_size),  # pad to 1000 * 1000
        ]
    )


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU"""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 1, 3, 1, 1)
        # assert max(std) <= 1 and min(std) >= 0\
        #     or max(mean) <= 1 and min(mean) >= 0,\
        #         "Please provide mean or std within range [0, 1]"

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (B, N, 3, H, W)

        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.0)
        return img.sub_(self.mean).div_(self.std)


def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    """
    Args:
        examples: iterable, examples grouped by image/video
        chunk_size: int, number of examples in each chunk.
        pad_to_divisible: bool, pad the examples to be divisible by chunk_size.
    >>> test_examples = [3, 4, 5, 6, 7]
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=True)
    [[3, 4], [5, 6], [7, 7]]  # the lst element has some randomness
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=False)
    [[3, 4], [5, 6], [7]]
    """
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i * chunk_size : (i + 1) * chunk_size])
    return chunked_examples


def mk_input_group(
    key_grouped_examples,
    max_n_example_per_group=1,
    is_train=True,
    example_unique_key=None,
):
    """Re-organize examples into groups. Each input group will have a single image paired
    with X (X=max_n_example_per_img) examples. Images with total #examples > X will be
    split into multiple groups. In the case a group has < X examples, we will copy
    the examples to make the group has X examples.
    Args:
        key_grouped_examples: dict, each key is image/video id,
            each value is a list(example) associated with this image/video
        max_n_example_per_group: int, pair max #examples with each image/video.
           Note that each image can have multiple groups.
        is_train: bool, if True, copy the examples to make sure each input
            group has max_n_example_per_group examples.
        example_unique_key: str, used to make sure no inputs are discarded by matching
            the input and output ids specified by `example_unique_key`
    """
    input_groups = []  # each element is (id, list(example))
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(
            examples,
            chunk_size=max_n_example_per_group,
            pad_to_divisible=is_train,
        )
        for c in chunked_examples:
            # if len(c) == 0:
            #     continue
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(
            f"Using example_unique_key {example_unique_key} to check whether input and output ids m"
        )
        # sanity check: make sure we did not discard any input example by accident.
        input_question_ids = flat_list_of_lists(
            [
                [sub_e[example_unique_key] for sub_e in e]
                for e in key_grouped_examples.values()
            ]
        )
        output_question_ids = flat_list_of_lists(
            [
                [sub_e[example_unique_key] for sub_e in e[1]]
                for e in input_groups
            ]
        )
        assert set(input_question_ids) == set(
            output_question_ids
        ), "You are missing "
    return input_groups


# def repeat_tensor_rows(raw_tensor, row_repeats):
#     """ repeat raw_tensor[i] row_repeats[i] times.
#     Args:
#         raw_tensor: (B, *)
#         row_repeats: list(int), len(row_repeats) == len(raw_tensor)
#     """
#     assert len(raw_tensor) == len(raw_tensor), "Has to be the same length"
#     if sum(row_repeats) == len(row_repeats):
#         return raw_tensor
#     else:
#         indices = torch.LongTensor(
#             flat_list_of_lists([[i] * r for i, r in enumerate(row_repeats)])
#         ).to(raw_tensor.device)
#         return raw_tensor.index_select(0, indices)


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        "",
        question.lower(),
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


def save_result(result, result_dir, filename, remove_duplicate=""):
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, utils.get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)

    return final_result_file


# def coco_caption_eval(coco_gt_root, results_file, split):

#     if "coco" in coco_gt_root:
#         urls = {
#             "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
#             "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
#         }
#         filenames = {
#             "val": "coco_karpathy_val_gt.json",
#             "test": "coco_karpathy_test_gt.json",
#         }

#         download_url(urls[split], coco_gt_root)
#         annotation_file = os.path.join(coco_gt_root, filenames[split])

#     else:
#         annotation_file = coco_gt_root

#     # create coco object and coco_result object
#     coco = COCO(annotation_file)
#     coco_result = coco.loadRes(results_file)

#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)

#     # evaluate on a subset of images by setting
#     # coco_eval.params['image_id'] = coco_result.getImgIds()
#     # please remove this line when evaluating the full validation set
#     # coco_eval.params['image_id'] = coco_result.getImgIds()

#     # evaluate results
#     # SPICE will take a few minutes the first time, but speeds up due to caching
#     coco_eval.evaluate()

#     # print output evaluation scores
#     for metric, score in coco_eval.eval.items():
#         print(f"{metric}: {score:.3f}")

#     return coco_eval


def mappixels(img, logit_laplace_eps=0.1):
    img = torch.true_divide(img, 255)
    img = (1 - 2 * logit_laplace_eps) * img + logit_laplace_eps
    return img


def copyresize(img, scale=None, interpolation="lanczos"):
    if scale is not None and not isinstance(scale, tuple):
        scale = (scale,) * 2

    if scale is not None:
        img = mmcv.imresize(img, scale, interpolation=interpolation)

    return img


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

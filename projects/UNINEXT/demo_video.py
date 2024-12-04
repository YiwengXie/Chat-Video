import os
import math
import numpy as np
import cv2
import torch
import sqlite3
import ruamel.yaml as yaml
from PIL import Image
from transformers import AutoTokenizer
from detectron2.config import get_cfg

from detectron2.utils.visualizer import ColorMode
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.modeling import build_model
from projects.Omnivl.data.utils import load_video_from_path_decord
from .uninext.data.coco_dataset_mapper_uni import (
    create_queries_and_maps,
)

from .uninext.data.datasets.bdd100k import (
    BDD_TRACK_CATEGORIES,
)
from .uninext.data.datasets.ytvis import (
    OVIS_CATEGORIES,
    YTVIS_CATEGORIES_2019,
    YTVIS_CATEGORIES_2021,
)
from .uninext.data.ytvis_eval import (
    instances_to_coco_json_video,
)
from projects.Omnivl.demo_omnivl_video import OmniVL_VideoDemo
from projects.Omnivl.demo_omnivl_image import OmniVLVQAPredictor
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from projects.BLIP2.demo_blip2_caption import select_device

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

The records in the tables are in the following format:

```
ID: the primary key of the record
Category: the category of the object
Appearance: the appearance of the object
Motion: the motion of the object
Trajectory: the trajectory of the object
Velocity: the velocity of the object
```

The records in the tables are randomly ordered.

If the results of the SQLQuery are empty, try to retrieve more information from the database to answer the question. You could try up to 3 times, and if all the results are empty, you could finish the chain.

If the results of the SQLQuery include multiple records, you should list them separately in your answers instead of mixing them together.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)

from detectron2.config import CfgNode as CN
def add_uninext_config(cfg):
    """
    Add config for UNINEXT.
    """
    # Unification of detection & grounding
    cfg.UNI = True  # Unified detection & grounding
    cfg.UNI_VID = False  # Unified video tasks joint training
    cfg.MODEL.DECOUPLE_TGT = False  # detection and grounding use different tgt (nn.Embedding vs Language)
    cfg.MODEL.STILL_TGT_FOR_BOTH = (
        False  # both detection and grounding use still (learnable) tgt
    )
    cfg.MODEL.CLS_POOL_TYPE = "average"  # average, max
    cfg.MODEL.USE_IOU_BRANCH = False  # add an IoU branch parallel to cls head
    cfg.MODEL.PARALLEL_DET = False  # parallel formulation for object detection
    cfg.MODEL.OTA = False
    cfg.MODEL.FREEZE_TEXT_ENCODER = False  # freeze the text encoder

    # ReID head
    cfg.DETACH_REID = False  # whether to detach reid
    cfg.USE_DEFORMABLE_REID_HEAD = False
    cfg.N_LAYER_DEFORMABLE_REID = 2

    cfg.DATASETS.TRAIN = []  # replace tuple with List

    # Unified dataloader for multiple tasks
    # cfg.DATALOADER.SAMPLER_TRAIN = "MultiDatasetSampler"
    cfg.DATALOADER.DATASET_RATIO = [1, 1]
    cfg.DATALOADER.USE_DIFF_BS_SIZE = True
    cfg.DATALOADER.DATASET_BS = [2, 2]
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.DATASET_ANN = ["box", "image"]

    # Allow different datasets to use different input resolutions
    cfg.INPUT.MIN_SIZE_TRAIN_MULTI = [
        (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        (320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640),
    ]
    cfg.INPUT.MAX_SIZE_TRAIN_MULTI = [1333, 768]

    # BoxInst
    cfg.MODEL.BOXINST = CN()
    cfg.MODEL.BOXINST.ENABLED = False  # Whether to enable BoxInst
    cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10
    cfg.MODEL.BOXINST.PAIRWISE = CN()
    cfg.MODEL.BOXINST.PAIRWISE.SIZE = 3
    cfg.MODEL.BOXINST.PAIRWISE.DILATION = 2
    cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
    cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3
    cfg.MODEL.BOXINST.TOPK = (
        64  # max number of proposals for computing mask loss
    )

    # MOT & MOTS thresholds
    cfg.TRACK = CN()
    cfg.TRACK.INIT_SCORE_THR = 0.5  # score threshold to start a new track
    cfg.TRACK.ADDNEW_SCORE_THR = 0.5  # score threshold to add a new track
    cfg.TRACK.MATCH_SCORE_THR = 0.5  # score threshold to match a track
    cfg.TRACK.OBJ_SCORE_THR = 0.3  # score threshold to continue a track

    # SOT & VOS
    cfg.SOT = CN()
    cfg.SOT.TEMPLATE_SZ = 256
    cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE = False
    cfg.SOT.SEARCH_AREA_FACTOR = 2
    cfg.SOT.REF_FEAT_SZ = 8  # resize to (REF_FEAT_SZ, REF_FEAT_SZ)
    cfg.SOT.FEAT_FUSE = False  # SOT feature fusion among P3~P6
    cfg.SOT.ONLINE_UPDATE = (
        False  # whether to adopt template update during inference
    )
    cfg.SOT.UPDATE_INTERVAL = 200
    cfg.SOT.UPDATE_THR = 0.7
    # VOS inference
    cfg.SOT.INFERENCE_ON_3F = False
    cfg.SOT.INST_THR_VOS = (
        0.5  # if the instance score < INST_THR_VOS, return a blank mask
    )

    cfg.MODEL.LANG_GUIDE_DET = (
        True  # Language-guided detection (similar to GLIP)
    )
    cfg.MODEL.VL_FUSION_USE_CHECKPOINT = (
        True  # Use gradient checkpoint for VL Early Fusion
    )
    cfg.MODEL.USE_EARLY_FUSION = (
        True  # Use Early Fusion (Bidirectional Cross-Modal Attention)
    )
    cfg.MODEL.USE_ADDITIONAL_BERT = (
        False  # Use additional BERT Layers in early fusion
    )
    cfg.MODEL.LANG_AS_CLASSIFIER = True  # Use language embedding as classifier
    cfg.MODEL.STILL_CLS_FOR_ENCODER = False  # Use still classifier for encoder

    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
    cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = (
        256  # max length of the tokenized captions.
    )
    cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
    cfg.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
    cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False
    cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True

    cfg.MODEL.DYHEAD = CN()
    cfg.MODEL.DYHEAD.PRIOR_PROB = 0.01
    cfg.MODEL.DYHEAD.LOG_SCALE = 0.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG = CN()
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = True

    # DataLoader
    cfg.INPUT.DATASET_MAPPER_NAME = (
        "detr"  # use "coco_instance_lsj" for LSJ aug
    )
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.IMAGE_SIZE_LARGE = 1024  # Larger input size (1536)
    # mixup
    cfg.INPUT.USE_MIXUP = False
    cfg.INPUT.MIXUP_PROB = 1.0

    # Video Sampler
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10  # 10 frames for VIS, R-VOS
    cfg.INPUT.SAMPLING_FRAME_RANGE_MOT = 3  # 3 frames for BDD100K
    cfg.INPUT.SAMPLING_FRAME_RANGE_SOT = 200  # 200 frames for SOT datasets
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = (
        []
    )  # "brightness", "contrast", "saturation", "rotation"

    # VIS Evaluation
    cfg.MODEL.IDOL = CN()
    cfg.MODEL.IDOL.CLIP_STRIDE = 1
    cfg.MODEL.IDOL.MERGE_ON_CPU = True
    cfg.MODEL.IDOL.MULTI_CLS_ON = True
    cfg.MODEL.IDOL.APPLY_CLS_THRES = 0.05
    cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE = (
        "mean"  # mean or max score for sequence masks during inference,
    )
    cfg.MODEL.IDOL.INFERENCE_SELECT_THRES = 0.1  # 0.05 for ytvis
    cfg.MODEL.IDOL.INFERENCE_FW = True  # frame weight
    cfg.MODEL.IDOL.INFERENCE_TW = True  # temporal weight
    cfg.MODEL.IDOL.MEMORY_LEN = 3
    cfg.MODEL.IDOL.BATCH_INFER_LEN = 10

    cfg.MODEL.DDETRS = CN()
    cfg.MODEL.DDETRS.FREEZE_ATTN = False
    cfg.MODEL.DDETRS.NUM_CLASSES = None
    cfg.MODEL.DDETRS.USE_CHECKPOINT = (
        False  # whether to use gradient-checkpoint for the transformer
    )
    # LOSS
    cfg.MODEL.DDETRS.MASK_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DICE_WEIGHT = 5.0
    cfg.MODEL.DDETRS.GIOU_WEIGHT = 2.0
    cfg.MODEL.DDETRS.L1_WEIGHT = 5.0
    cfg.MODEL.DDETRS.CLASS_WEIGHT = 2.0
    cfg.MODEL.DDETRS.REID_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DEEP_SUPERVISION = True
    cfg.MODEL.DDETRS.MASK_STRIDE = 4
    cfg.MODEL.DDETRS.MATCH_STRIDE = 4
    cfg.MODEL.DDETRS.FOCAL_ALPHA = 0.25
    # COST
    cfg.MODEL.DDETRS.SET_COST_CLASS = 2
    cfg.MODEL.DDETRS.SET_COST_BOX = 5
    cfg.MODEL.DDETRS.SET_COST_GIOU = 2
    # TRANSFORMER
    cfg.MODEL.DDETRS.NHEADS = 8
    cfg.MODEL.DDETRS.DROPOUT = 0.1
    cfg.MODEL.DDETRS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DDETRS.ENC_LAYERS = 6
    cfg.MODEL.DDETRS.DEC_LAYERS = 6
    cfg.MODEL.DDETRS.NUM_VL_LAYERS = 1  # one layer for early fusion is enough
    cfg.MODEL.DDETRS.VL_HIDDEN_DIM = 2048  # embed_dim of BiAttentionBlock
    cfg.MODEL.DDETRS.TWO_STAGE = False
    cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS = 300
    cfg.MODEL.DDETRS.MIXED_SELECTION = False
    cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE = False
    cfg.MODEL.DDETRS.CTRL_LAYERS = 3
    cfg.MODEL.DDETRS.USE_DINO = False
    cfg.MODEL.DDETRS.USE_BLIP2DINO = False
    cfg.MODEL.DDETRS.DYNAMIC_LABEL_ENC = False
    cfg.MODEL.DDETRS.HIDDEN_DIM = 256
    cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.DDETRS.DEC_N_POINTS = 4
    cfg.MODEL.DDETRS.ENC_N_POINTS = 4
    cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS = 4
    # Mask Postprocessing & Upsampling
    cfg.MODEL.DDETRS.MASK_THRES = 0.5
    cfg.MODEL.DDETRS.NEW_MASK_HEAD = False
    cfg.MODEL.DDETRS.USE_RAFT = False
    cfg.MODEL.DDETRS.USE_REL_COORD = True
    # Denoising
    cfg.MODEL.DDETRS.DN_NUMBER = 100
    cfg.MODEL.DDETRS.LABEL_NOISE_RATIO = 0.5
    cfg.MODEL.DDETRS.BOX_NOISE_SCALE = 1.0

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.LINEAR_PROJ_MULTIPLIER = 0.1
    cfg.SOLVER.LANG_LR = 0.00001  # 1e-5
    cfg.SOLVER.VL_LR = 0.00001  # 1e-5

    cfg.SOLVER.LOSS_WEIGHT_LM = 1.0
    cfg.SOLVER.LOSS_WEIGHT_DET = 1.0
    cfg.SOLVER.LOSS_WEIGHT_GRD = 1.0
    cfg.SOLVER.LOSS_WEIGHT_SOT = 1.0

    # R50 backbone
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
    # supprt ConvNeXt backbone
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.NAME = "tiny"
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.7
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False
    # supprt ViT backbone
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.NAME = "ViT-Base"
    cfg.MODEL.VIT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.VIT.USE_CHECKPOINT = False

    cfg.FIND_UNUSED_PARAMETERS = False  # find_unused_parameters
    cfg.TEST.EVAL_AFTER_TRAIN = True  # eval after train


class Tracklet:
    def __init__(
        self, id, category, appearance=None, motion=None, trajectory=[]
    ):
        self.id = id
        self.appearance = appearance
        self.motion = motion
        self.trajectory = trajectory
        self.category = category

    def set_appearance(self, appearance):
        self.appearance = appearance

    def set_motion(self, motion):
        self.motion = motion

    def add_trajectory(self, time, coordinates):
        self.trajectory.append((time, coordinates))

    def clear_trajectory(self):
        self.trajectory = []

    def generate_description(self, trajectory_only=False):
        # assert len(self.trajectory) > 0
        if self.appearance is None:
            self.appearance = "the appearance is unknown"
        if self.motion is None:
            self.motion = "the motion is unknown"

        if not trajectory_only:
            des = f"{self.id}th instance, {self.category}, {self.appearance}, {self.motion}, the trajectory is (coordinate represented in the form of (x1, y1, x2, y2)): "
        else:
            des = "the trajectory is (coordinate represented in the form of (x1, y1, x2, y2)): "

        for t, c in self.trajectory[:2]:
            des += f"at {t} seconds, ({int(c[0])},{int(c[1])},{int(c[2])},{int(c[3])}), "

        des = des[:-2] + "..."

        return des


def calculate_velocity(first_box, last_box, t):
    if t == 0:
        return 0
    else:
        width = (first_box[0] + first_box[2] - last_box[0] - last_box[2]) / 2
        height = (first_box[1] + first_box[3] - last_box[1] - last_box[3]) / 2
        distance = (height ** 2 + width ** 2) ** 0.5
        return distance / t


def build_database(all_tracklets, video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    sql_path = os.path.join(video_dir, video_name + ".db")
    if os.path.exists(sql_path):
        os.remove(sql_path)
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS tracklets (id INTEGER PRIMARY KEY, category TEXT, appearance TEXT, motion TEXT, trajectory TEXT, velocity REAL)"
    )
    conn.commit()
    for track in all_tracklets:
        track_id = track.id
        category = track.category
        appearance = track.appearance.replace("'", "")
        motion = track.motion if track.motion is not None else "unknown"
        motion = track.motion.replace("'", "")
        trajectory = track.generate_description(trajectory_only=True)
        velocity = calculate_velocity(
            track.trajectory[0][1],
            track.trajectory[-1][1],
            len(track.trajectory),
        )
        command = f"INSERT INTO tracklets (id, category, appearance, motion, trajectory, velocity) \
                  VALUES ({track_id}, '{category}', '{appearance}', '{motion}', '{trajectory}', '{velocity}')"
        print(command)
        c.execute(command)
        conn.commit()

    conn.close()
    return sql_path


def setup_cfg(config_file, weights):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_uninext_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.IDOL.MULTI_CLS_ON = False
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.USE_IOU_BRANCH = False
    cfg.freeze()
    return cfg


class VisualizationDemo(object):
    def __init__(
        self,
        cfg,
        dataset,
        openai_key,
        max_frames=100,
        instance_mode=ColorMode.IMAGE,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.instance_mode = instance_mode
        self.predictor = UNINEXTPredictor(cfg, dataset=dataset)
        self.max_frames = max_frames
        self.test_dataset = dataset
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=openai_key,
        )

    def set_maxframes(self, max_frames):
        self.max_frames = max_frames
        print("max frames set to ", self.max_frames)

    def _frame_from_video(self, video_path, start_time=0, end_time=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        true_video_length = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

        count = 0
        frames = []
        file_names = []
        if end_time is None:
            end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                count += 1
                if count > (start_time * fps) and count <= (end_time * fps):
                    frames.append(frame)
                    if self.test_dataset == "vis19":
                        file_names.append(os.path.join("dataset", "ytvis_2019"))
                    else:
                        file_names.append(os.path.join("dataset", "coco"))

                if count == (end_time * fps):
                    break

            else:
                break

        total_frames = len(frames)
        max_frames_to_track = self.max_frames

        # print("Total frames: ", total_frames, max_frames_to_track)
        if total_frames > max_frames_to_track:
            step = total_frames // max_frames_to_track
            frames = frames[::step]
            file_names = file_names[::step]
        else:
            step = 1

        # print(len(frames), step, true_video_length)
        return frames, file_names, fps, true_video_length, step

    def run_on_video(self, video, question):
        video_dir = os.path.dirname(video)
        video_name = os.path.basename(video).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + ".db")

        # when you click th tracking button, or want to ask a question, but the tracklet is already there
        if os.path.exists(sql_path):
            print("Tracklet already exists")
            frames_read = filenames_read = None
            fps = step = 1
            video_length = 0
            height = width = 0
            model_outputs = None

        # tracklet does not exist, so we need to track it
        else:
            (
                frames_read,
                filenames_read,
                fps,
                video_length,
                step,
            ) = self._frame_from_video(video)

            all_tracklets, model_outputs = self.predictor(
                video,
                frames_read,
                filenames_read,
                fps,
                step,
            )
            print('\n\n\n\n\n\n\n\n\n\n', model_outputs)
            height, width = frames_read[0].shape[:2]
            sql_path = build_database(all_tracklets, video)

        db = SQLDatabase.from_uri("sqlite:///" + sql_path)
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, prompt=PROMPT, verbose=True
        )
        if len(question) > 0:
            re = db_chain.run(question)
        else:
            re = ""

        video_info = {
            "frames": frames_read,
            "fps": fps / step,
            "video_length": video_length,
            "height": height,
            "width": width,
        }
        return re, model_outputs, video_info


class UNINEXTPredictor:
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

    def __init__(self, cfg, dataset):
        # selected_device = select_device()
        selected_device = "cuda:0"
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.DEVICE = selected_device
        self.cfg.freeze()
        print("Creating UNINEXT on device: ", selected_device)
        self.device = torch.device(selected_device)
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # Optionaly, you could use BLIP2 to generate the appearance caption
        # self.captioner = BLIP2Predictor(
        #     tag="OPT6.7B", bit8=True, device=selected_device
        # )
        config_path = "projects/Omnivl/configs/share/downstream/vqa.yaml"
        config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
        config["pretrained"] = "pretrained_models/vqa.pth"
        self.captioner = OmniVLVQAPredictor(config, device=selected_device)

        self.dynamic_captioner = OmniVL_VideoDemo(frames_per_segment=256)

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        # for UNINEXT
        self.tokenizer = AutoTokenizer.from_pretrained(
            "projects/UNINEXT/bert-base-uncased"
        )
        self.test_dataset = dataset
        self.prompt_test_dict = {}
        self.positive_map_label_to_token_dict = {}

        for dataset_name in [
            "ytvis_2019",
            "ytvis_2021",
            "ytvis_ovis",
            "coco",
            "bdd_box_track",
            "bdd_seg_track",
        ]:
            if dataset_name.startswith("ytvis_2019"):
                (
                    prompt_test,
                    positive_map_label_to_token,
                ) = create_queries_and_maps(
                    YTVIS_CATEGORIES_2019, self.tokenizer
                )
                self.prompt_test_dict["vis19"] = prompt_test
                self.positive_map_label_to_token_dict[
                    "vis19"
                ] = positive_map_label_to_token
            elif dataset_name.startswith("ytvis_2021"):
                (
                    prompt_test,
                    positive_map_label_to_token,
                ) = create_queries_and_maps(
                    YTVIS_CATEGORIES_2021, self.tokenizer
                )
                self.prompt_test_dict["vis21"] = prompt_test
                self.positive_map_label_to_token_dict[
                    "vis21"
                ] = positive_map_label_to_token
            elif dataset_name.startswith("ytvis_ovis"):
                (
                    prompt_test,
                    positive_map_label_to_token,
                ) = create_queries_and_maps(OVIS_CATEGORIES, self.tokenizer)
                self.prompt_test_dict["ovis"] = prompt_test
                self.positive_map_label_to_token_dict[
                    "ovis"
                ] = positive_map_label_to_token
            elif dataset_name.startswith("coco"):
                (
                    prompt_test,
                    positive_map_label_to_token,
                ) = create_queries_and_maps(COCO_CATEGORIES, self.tokenizer)
                self.prompt_test_dict["coco"] = prompt_test
                self.positive_map_label_to_token_dict[
                    "coco"
                ] = positive_map_label_to_token

            elif dataset_name.startswith(
                "bdd_box_track"
            ) or dataset_name.startswith("bdd_seg_track"):
                (
                    prompt_test,
                    positive_map_label_to_token,
                ) = create_queries_and_maps(
                    BDD_TRACK_CATEGORIES, self.tokenizer
                )
                self.prompt_test_dict["bdd_track"] = prompt_test
                self.positive_map_label_to_token_dict[
                    "bdd_track"
                ] = positive_map_label_to_token

    def get_roi(self, frame, box=None):
        if box is None:
            masked_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masked_frame_pil = Image.fromarray(masked_frame)

        else:
            x1, y1, x2, y2 = box
            masked_frame = np.zeros(frame.shape, dtype=np.uint8)
            masked_frame[int(y1) : int(y2), int(x1) : int(x2)] = frame[
                int(y1) : int(y2), int(x1) : int(x2)
            ]
            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
            masked_frame_pil = Image.fromarray(masked_frame)

        return masked_frame_pil

    def get_roi_for_decord(self, frame, box=None):
        if box is None:
            masked_frame = frame

        else:
            x1, y1, x2, y2 = box
            height, width = (y2 - y1), (x2 - x1)
            x1 = max(0, x1 - width)
            y1 = max(0, y1 - height)
            x2 = min(frame.shape[1], x2 + width)
            y2 = min(frame.shape[0], y2 + height)
            masked_frame = np.zeros(frame.shape, dtype=np.uint8)
            masked_frame[int(y1) : int(y2), int(x1) : int(x2)] = frame[
                int(y1) : int(y2), int(x1) : int(x2)
            ]
        return masked_frame

    def distance_to_bound(self, frame, box):
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        area = (x2 - x1) * (y2 - y1)
        return min(x1, y1, width - x2, height - y2) + math.sqrt(area)

    def dynamic_des(self, track_rois, category):
        """
        frames: list of frames
        trajectories: list of trajectories
        """
        vid_length = len(track_rois)
        frames_per_segment = self.dynamic_captioner.frames_per_segment
        num_segments = math.ceil(vid_length / frames_per_segment)

        segments = []
        time_intervals = []
        for i in range(num_segments):
            start = i * frames_per_segment
            end = (i + 1) * frames_per_segment
            if end > vid_length:
                end = vid_length

            time_inter = [track_rois[i][0] for i in range(start, end)]
            seg = [track_rois[i][1] for i in range(start, end)]

            seg = np.stack(seg, axis=0)
            segments.append(seg)
            time_intervals.append(time_inter)

        object_dynamics = self.dynamic_captioner.predictor(
            segments, customized_prompt=""
        )

        object_motion_desp = ""
        for time_inter, dyn in zip(time_intervals, object_dynamics):
            start_time = time_inter[0]
            end_time = time_inter[-1]
            if category == "environment":
                caption = (
                    f"From {start_time} seconds to {end_time} seconds, {dyn}"
                )
            else:
                caption = f"From {start_time} seconds to {end_time} seconds, the {category} is {dyn}"
            object_motion_desp += caption + "; "

        return object_motion_desp[:-2]

    def __call__(
        self,
        video_path,
        original_images,
        file_names,
        fps,
        step,
        task="detection",
        expressions=None,
    ):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        # T H W C
        vid_frm_array, _ = load_video_from_path_decord(
            video_path,
            "all",
            num_frm=-1,  # when num_frm=-1, it will load all frames
            return_fps=True,
        )
        vid_frm_array = vid_frm_array[::step]

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_images[0].shape[:2]
            images = []
            for original_image in original_images:
                image = self.aug.get_transform(original_image).apply_image(
                    original_image
                )
                image = torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1)
                )  # .to(self.device)
                images.append(image)

            # for UNINEXT
            dataset_name = "vis19"
            expressions = self.prompt_test_dict[self.test_dataset]
            positive_map_label_to_token = self.positive_map_label_to_token_dict[
                self.test_dataset
            ]

            inputs = {
                "image": images,
                "height": height,
                "width": width,
                "task": task,
                "expressions": [expressions],
                "file_names": file_names,
                "dataset_name": dataset_name,
                "video_id": 0,
                "length": len(images),
                "positive_map_label_to_token": positive_map_label_to_token,
            }

            outputs = self.model([inputs])
            print("outputs", outputs)
            _, predictions, _ = instances_to_coco_json_video([inputs], outputs)

            all_tracklets = []

            for k, v in predictions.items():
                tracklet = Tracklet(id=k, category=v["category"])
                tracklet.clear_trajectory()
                distances = []
                track_rois = []
                for frame_id, box in enumerate(v["boxes"]):
                    t = round(frame_id * step / fps, 1)
                    if box is not None:
                        tracklet.trajectory.append(
                            (t, box.detach().cpu().numpy().tolist())
                        )
                        distances.append(
                            self.distance_to_bound(
                                original_images[frame_id], box
                            )
                        )
                        roi = self.get_roi_for_decord(
                            vid_frm_array[frame_id],
                            box.detach().cpu().numpy().tolist(),
                        )
                        track_rois.append((t, roi))
                    else:
                        distances.append(-1)

                frame_with_max_dist = np.argmax(distances)
                object_region = self.get_roi(
                    original_images[frame_with_max_dist],
                    v["boxes"][frame_with_max_dist]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                )

                if v["category"] == "person":
                    question = "Describe the appearance and clothing of the person in detail: the person"
                else:
                    question = f"Describe the appearance of the {v['category']} in detail: the {v['category']}"

                object_attributes = self.captioner.ask(object_region, question)
                object_attributes = (
                    v["category"]
                    + " "
                    + object_attributes.replace("\n", "").lower()
                )

                object_motion = self.dynamic_des(track_rois, v["category"])
                tracklet.set_motion(object_motion)
                tracklet.set_appearance(object_attributes)
                all_tracklets.append(tracklet)

            return all_tracklets, outputs

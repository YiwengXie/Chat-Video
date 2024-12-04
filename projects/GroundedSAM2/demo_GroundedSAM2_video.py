import os
import math
import numpy as np
import cv2
from sympy import fps
import torch
import sqlite3
import json
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
from ..UNINEXT.uninext.data.coco_dataset_mapper_uni import (
    create_queries_and_maps,
)
from ..UNINEXT.uninext.data.datasets.bdd100k import (
    BDD_TRACK_CATEGORIES,
)
from ..UNINEXT.uninext.data.datasets.ytvis import (
    OVIS_CATEGORIES,
    YTVIS_CATEGORIES_2019,
    YTVIS_CATEGORIES_2021,
)
from ..UNINEXT.uninext.data.ytvis_eval import (
    instances_to_coco_json_video,
)
from projects.LLaVA_NeXT.demo_LLaVA_video import LLaVAVideoDescriber
from projects.LLaVA_NeXT.demo_LLaVA_image import LLaVADescriber
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from projects.BLIP2.demo_blip2_caption import select_device
from projects.GroundedSAM2.demo_GroundedSAM2_tracker import VideoObjectTracker

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
        if len(track.trajectory) > 0:
            velocity = calculate_velocity(
                track.trajectory[0][1],
                track.trajectory[-1][1],
                len(track.trajectory),
            )
        else:
            velocity = 0
        command = f"INSERT INTO tracklets (id, category, appearance, motion, trajectory, velocity) \
                  VALUES ({track_id}, '{category}', '{appearance}', '{motion}', '{trajectory}', '{velocity}')"
        print(command)
        c.execute(command)
        conn.commit()

    conn.close()
    return sql_path




class VisualizationDemo(object):
    def __init__(
        self,
        dataset,
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
        self.instance_mode = instance_mode
        self.predictor = GroundedSAM2Predictor(dataset=dataset)
        self.max_frames = max_frames
        self.test_dataset = dataset
        self.fps_set = 0
        self.llm = OpenAI(temperature=0,
                          openai_api_key=os.getenv("OPENAI_API_KEY"),
                          model="gpt-3.5-turbo-instruct"
                          )
        
    def set_maxframes(self, max_frames):
        self.max_frames = max_frames
        print("max frames set to ", self.max_frames)

    def set_fps(self, fps):
        self.fps_set = fps
        print("fps set to ", self.fps_set)

    def _frame_from_video(self, video_path, output_folder, start_time=0, end_time=None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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


        fps_set_step = 1
        maxframes_step = 1

        if self.fps_set != 0:
            fps_set_step = max(1, math.ceil(fps / self.fps_set))
            frames = frames[::fps_set_step]
            file_names = file_names[::fps_set_step]


        total_frames = len(frames)
        max_frames_to_track = self.max_frames

        # print("Total frames: ", total_frames, max_frames_to_track)
        if total_frames > max_frames_to_track:
            maxframes_step = total_frames // max_frames_to_track
            frames = frames[::maxframes_step]
            file_names = file_names[::maxframes_step]
        
        step = fps_set_step * maxframes_step

        # Saving every `step` frame
        for idx, frame in enumerate(frames):
            file_name = f"{idx}.jpg"
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, frame)

        # print(len(frames), step, true_video_length)
        return frames, file_names, fps, true_video_length, step

    def run_on_video(self, video, question):
        video_dir = os.path.dirname(video)
        video_name = os.path.basename(video).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + ".db")

        frames_dir = "videos/tracking_frames"

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
            ) = self._frame_from_video(video, output_folder=frames_dir)

            all_tracklets, model_outputs = self.predictor(
                video,
                frames_read,
                frames_dir,
                filenames_read,
                fps,
                step,
            )

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

        new_re_index = re.find('Question:')
        if new_re_index != -1:
            re = re[:new_re_index].strip()
        else:
            re = re

        video_info = {
            "frames": frames_read,
            "fps": fps,
            "video_length": video_length,
            "height": height,
            "width": width,
            "step": step,
        }
        return re, model_outputs, video_info


class GroundedSAM2Predictor:

    def __init__(self, dataset):
        # selected_device = select_device()
        selected_device = "cuda:0"
        print("Creating Grounded-SAM2 on device: ", selected_device)
        self.device = torch.device(selected_device)
        self.test_dataset = dataset
        self.prompt_test_dict = {}
        self.positive_map_label_to_token_dict = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "projects/UNINEXT/bert-base-uncased"
        )
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
        self.expression = self.prompt_test_dict[dataset]

        self.model = VideoObjectTracker(self.expression)
        # Optionaly, you could use BLIP2 to generate the appearance caption
        # self.captioner = BLIP2Predictor(
        #     tag="OPT6.7B", bit8=True, device=selected_device
        # )
        config_path = "projects/Omnivl/configs/share/downstream/vqa.yaml"
        config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
        config["pretrained"] = "pretrained_models/vqa.pth"
        self.captioner = LLaVADescriber(device=selected_device)
        self.dynamic_captioner = LLaVAVideoDescriber(device=selected_device, max_frames=16)
        

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
        frames_per_segment = self.dynamic_captioner.max_frames
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

        # object_dynamics = self.dynamic_captioner.predictor(
        #     segments, customized_prompt=""
        # )
        if category == "environment":
            question = "Describe what's happening in this video in detail."
        else:
            question = f"Describe what's the motion and behavior of the {category} in this video in detail."
        object_dynamics = [self.dynamic_captioner.generate(seg, question, type='video_array') for seg in segments]

        object_motion_desp = ""
        for time_inter, dyn in zip(time_intervals, object_dynamics):
            start_time = time_inter[0]
            end_time = time_inter[-1]
            if category == "environment":
                caption = (
                    f"From {start_time} seconds to {end_time} seconds, {dyn}"
                )
            else:
                caption = f"From {start_time} seconds to {end_time} seconds, {dyn}"
            object_motion_desp += caption + "; "

        return object_motion_desp[:-2]

    def __call__(
        self,
        video_path,
        original_images,
        frames_dir,
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
            import time
            start_time = time.time()
            output_bbox_json_path, outputs = self.model(frames_dir)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open("videos/time.txt", "w") as f:
                f.write(f"tracking time: {elapsed_time} seconds\n")

            all_tracklets = []

            # # add environment
            # tracklet = Tracklet(id=0, category="environment")
            # tracklet.clear_trajectory()
            # object_attributes = self.dynamic_captioner.generate(video_path, "brief describe what's the environment in this video", type="video_path")
            # print('object_attributes:', object_attributes)
            # track_rois = []
            # for i in range(len(vid_frm_array)):
            #     roi = self.get_roi(vid_frm_array[i])
            #     t = round(i * step / fps, 1)
            #     track_rois.append((t, roi))
            # object_motion = self.dynamic_des(track_rois, "environment")
            # tracklet.set_motion(object_motion)
            # tracklet.set_appearance(object_attributes)
            # all_tracklets.append(tracklet)

            # add objects
            start_time = time.time()
            with open(output_bbox_json_path, "r") as file:
                data = json.load(file)
                instances = data["instances"]

                for instance_id, instance_info in instances.items():
                    category = instance_info["category"]
                    bboxes = instance_info["bboxes"]
                    tracklet = Tracklet(id=instance_id, category=category)
                    tracklet.clear_trajectory()
                    distances = []
                    track_rois = []

                    for bbox_info in bboxes:
                        frame_name = bbox_info.get('frame_name', '')
                        bbox = bbox_info.get('bbox', None)

                        if bbox is not None:
                            frame_id = int(frame_name.split('.')[0])
                            t = round(frame_id * step / fps, 1)
                            tracklet.add_trajectory(t, bbox)
                            distances.append(
                                self.distance_to_bound(
                                    original_images[frame_id], bbox
                                )
                            )
                            roi = self.get_roi_for_decord(
                                vid_frm_array[frame_id],
                                bbox,
                            )
                            track_rois.append((t, roi))

                        else:
                            distances.append(-1)
                    
                    frame_with_max_dist = np.argmax(distances)
                    object_region = self.get_roi(
                        original_images[frame_with_max_dist],
                        bboxes[frame_with_max_dist]['bbox'],
                    )

                    if category == "person":
                        question = "Describe the appearance and clothing of the person in detail."
                    else:
                        question = f"Describe the appearance of the {category} in detail."

                    # object_attributes = self.captioner.ask(object_region, question)
                    # object_attributes = (
                    #     category
                    #     + " "
                    #     + object_attributes.replace("\n", "").lower()
                    # )
                    object_attributes = self.captioner.generate(object_region, question)
                    object_motion = self.dynamic_des(track_rois, category)
                    tracklet.set_motion(object_motion)
                    tracklet.set_appearance(object_attributes)
                    all_tracklets.append(tracklet)

            end_time = time.time()
            elapsed_time = end_time - start_time
            with open("videos/time.txt", "a") as f:
                f.write(f"captioning time: {elapsed_time} seconds\n")

        return all_tracklets, outputs



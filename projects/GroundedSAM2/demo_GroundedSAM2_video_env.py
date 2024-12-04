import os
import math
from turtle import down
import numpy as np
import cv2
import librosa
import torch
import sqlite3
import ruamel.yaml as yaml
from PIL import Image
from torch.amp import autocast as autocast
from projects.LLaVA_NeXT.demo_LLaVA_video import LLaVAVideoDescriber
from projects.LLaVA_NeXT.demo_LLaVA_image import LLaVADescriber
from projects.BLIP2.demo_blip2_caption import select_device
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import torchaudio.transforms as transforms
from transformers import (
    pipeline,
    ASTFeatureExtractor,
    AutoModelForAudioClassification,
)
import whisper
from moviepy.editor import AudioFileClip
from IPython.display import Audio

emotion_model = pipeline(
    "audio-classification",
    # model="superb/wav2vec2-base-superb-er",
    model='projects/UNINEXT/wav2vec2-base-superb-er',
    device=0 if torch.cuda.is_available() else -1,
)

from langchain.prompts.prompt import PromptTemplate

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
id: the primary key of the record
appearance: the appearance information at different timestamps of the video
motion: the temporal dynamics and activities in the video
audio_cateogry: the category of the audio
audio_content: the content of the audio
audio_emotion: the emotion of the audio
```

The records in the tables are randomly ordered.

If the results of the SQLQuery are empty, try to retrieve more information from the database to answer the question. You could try up to 3 times, and if all the results are empty, you could finish the chain.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)


class Tracklet:
    def __init__(self, appearance=None, motion=None, audio=None):
        self.appearance = appearance
        self.motion = motion
        self.audio = audio

    def set_appearance(self, appearance):
        self.appearance = appearance

    def set_motion(self, motion):
        self.motion = motion

    def set_audio(self, category, content, emotion):
        self.audio = [category, content, emotion]


def build_database(tracklet, video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    sql_path = os.path.join(video_dir, video_name + "env.db")
    if os.path.exists(sql_path):
        os.remove(sql_path)
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS tracklets (id INTEGER PRIMARY KEY, appearance TEXT, motion TEXT, audio_category TEXT, audio_content TEXT, audio_emotion TEXT)"
    )
    conn.commit()
    appearance = tracklet.appearance.replace("'", "")
    motion = tracklet.motion.replace("'", "")
    tracklet.audio[1] = tracklet.audio[1].replace("'", "")
    command = f"INSERT INTO tracklets (id, appearance, motion, audio_category, audio_content, audio_emotion) \
        VALUES ({0}, '{appearance}', '{motion}', '{tracklet.audio[0]}', '{tracklet.audio[1]}', '{tracklet.audio[2]}')"

    print(command)
    c.execute(command)
    conn.commit()
    conn.close()
    return sql_path


class VisualizationEnvDemo(object):
    def __init__(self, max_frames=100):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        self.predictor = GroundedSAM2Predictor()
        self.max_frames = max_frames
        self.fps_set = 0

        selected_device = select_device()
        self.feature_extractor = ASTFeatureExtractor()
        self.ast_model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(selected_device)
        self.asr_model = whisper.load_model("base")

        self.emotion_mapping = {
            "neu": "neural",
            "hap": "happy",
            "sad": "sad",
            "ang": "angry",
        }
        self.device = selected_device
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

    def _frame_from_video(self, video_path, start_time=0, end_time=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        true_video_length = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

        count = 0
        frames = []
        if end_time is None:
            end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                count += 1
                if count > (start_time * fps) and count <= (end_time * fps):
                    frames.append(frame)

                if count == (end_time * fps):
                    break

            else:
                break

        fps_set_step = 1
        maxframes_step = 1

        if self.fps_set != 0:
            fps_set_step = max(1, math.ceil(fps / self.fps_set))
            frames = frames[::fps_set_step]
            
        total_frames = len(frames)
        max_frames_to_track = self.max_frames

        # print("Total frames: ", total_frames, max_frames_to_track)
        if total_frames > max_frames_to_track:
            maxframes_step = total_frames // max_frames_to_track
            frames = frames[::maxframes_step]

        step = fps_set_step * maxframes_step

        # print(len(frames), step, true_video_length)
        return frames, fps, true_video_length, step

    @autocast(device_type="cuda")
    def classification_ast(self, audio_path):
        waveform, sampling_rate = librosa.load(audio_path, sr=16000)
        Audio(waveform, rate=sampling_rate)
        # print(sampling_rate)
        if sampling_rate != 16000:
            transform = transforms.Resample(sampling_rate, 16000)
            waveform = transform(waveform)
            sampling_rate = 16000

        waveform = waveform
        with torch.no_grad():
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=sampling_rate,
                padding="max_length",
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(self.device)
            outputs = self.ast_model(input_values)

        predicted_class_idx = outputs.logits.argmax(-1).item()
        return self.ast_model.config.id2label[predicted_class_idx]

    def run_on_video(self, video, question):
        video_dir = os.path.dirname(video)
        video_name = os.path.basename(video).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + "env.db")
        if not os.path.exists(sql_path):
            (
                frames_read,
                fps,
                _,
                step,
            ) = self._frame_from_video(video)

            tracklet = self.predictor(
                video,
                frames_read,
                fps,
                step,
            )

            decord_audio = False
            try:
                my_audio_clip = AudioFileClip(video)
                audio_path = os.path.join(
                    os.path.dirname(video), "extract_audio.wav"
                )
                my_audio_clip.write_audiofile(audio_path)
                my_audio_clip.close()
                decord_audio = True

            except Exception as e:
                print(
                    "No audio in this video or fail to extract the audio: ", e
                )
                audio_description = (
                    "No audio in this video or fail to extract the audio."
                )

            if decord_audio:
                category = self.classification_ast(audio_path)

                with torch.no_grad():
                    text = self.asr_model.transcribe(audio_path)["text"]
                    """emotion = emotion_classifier(text)
                    detected_emotion = emotion[0]["label"]"""
                    emotion = emotion_model(audio_path, top_k=1)[0]["label"]
                    detected_emotion = self.emotion_mapping[emotion]

                tracklet.set_audio(category, text, detected_emotion)
                build_database(tracklet, video)

            else:
                tracklet.set_audio("No audio", "No audio", "No audio")
                build_database(tracklet, video)

        db = SQLDatabase.from_uri("sqlite:///" + sql_path)
        db_chain = SQLDatabaseChain(llm=self.llm, database=db, verbose=True)
        re = db_chain.run(question)

        return re


class GroundedSAM2Predictor:
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

    def __init__(self):
        selected_device = select_device()
        self.device = torch.device(selected_device)
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
            x1 = max(0, x1 - width // 2)
            y1 = max(0, y1 - height // 2)
            x2 = min(frame.shape[1], x2 + width // 2)
            y2 = min(frame.shape[0], y2 + height // 2)
            masked_frame = np.zeros(frame.shape, dtype=np.uint8)
            masked_frame[int(y1) : int(y2), int(x1) : int(x2)] = frame[
                int(y1) : int(y2), int(x1) : int(x2)
            ]
        return masked_frame

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
        appearances = []
        for i in range(num_segments):
            start = i * frames_per_segment
            end = (i + 1) * frames_per_segment
            if end > vid_length:
                end = vid_length

            time_inter = [track_rois[i][0] for i in range(start, end)]
            seg = [track_rois[j][1] for j in range(start, end)]
            # print(seg)
            seg = np.stack(seg, axis=0)
            segments.append(seg)
            time_intervals.append(time_inter)

        if category == "environment":
            question = "Describe what's the environment in this video in detail."
        else:
            question = f"Describe what's the appearence of the {category} in this video in detail."
        appearances = [self.dynamic_captioner.generate(seg, question, type='video_array') for seg in segments]

        if category == "environment":
            question = "Describe what's happening in this video in detail."
        else:
            question = f"Describe what's the motion and behavior of the {category} in this video in detail."

        object_dynamics = [self.dynamic_captioner.generate(seg, question, type='video_array') for seg in segments]


        object_motion_desp = ""
        object_app_desp = ""
        for time_inter, app, dyn in zip(
            time_intervals, appearances, object_dynamics
        ):
            start_time = time_inter[0]
            end_time = time_inter[-1]
            if category == "environment":
                caption = (
                    f"From {start_time} seconds to {end_time} seconds, {dyn}"
                )
            else:
                caption = f"From {start_time} seconds to {end_time} seconds, the {category} is {dyn}"
            object_motion_desp += caption + "; "
            object_app_desp += f"From {start_time} seconds to {end_time} seconds, {app}; "

        return object_motion_desp[:-2], object_app_desp[:-2]

    def __call__(
        self,
        video_path,
        original_images,
        fps,
        step,
    ):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        first_track = Tracklet()
        first_track_rois = []
        for frame_id, frm in enumerate(original_images):
            t = round(frame_id * step / fps, 1)
            frm = self.get_roi_for_decord(frm)
            first_track_rois.append((t, frm))

        video_motion, video_appearance = self.dynamic_des(
            first_track_rois, "environment"
        )
        first_track.set_motion(video_motion)
        first_track.set_appearance(video_appearance)
        return first_track


if __name__ == "__main__":
    demo = VisualizationEnvDemo(max_frames=2000)
    des = demo.run_on_video(
        "projects/GroundedSAM2/assets/zebra.mp4",
        "Describe the environment where the video takes place?",
    )
    print(des)

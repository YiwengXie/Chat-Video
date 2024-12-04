import os
import math
import torch
import ruamel.yaml as yaml
import spacy
import sqlite3
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from projects.Omnivl.models.blip import blip_decoder
from projects.Omnivl.data.utils import VideoNorm, load_video_from_path_decord
from projects.BLIP2.demo_blip2_caption import select_device


def add_escape(value):
    reserved_chars = r"""?&|!{}[]()^~*:\\"'+-"""
    replace = ["\\" + l for l in reserved_chars]
    trans = str.maketrans(dict(zip(reserved_chars, replace)))
    return value.translate(trans)


def build_database(
    all_cations, frames_per_segment, vid_length, fps, video_path
):

    """prompt = "Here is a video: (height: {}, width: {}, length: {}s). ".format(
        height, width, video_length
    )

    for tracklet in all_tracklets:
        prompt += tracklet.generate_description() + "."""

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    sql_path = os.path.join(video_dir, video_name + "omnivl.db")
    if os.path.exists(sql_path):
        os.remove(sql_path)
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS tracklets (id INTERGER PRIMARY KEY, start_time REAL, end_time REAL, caption TEXT)"
    )
    conn.commit()
    for seg_id, caption in enumerate(all_cations):
        id = seg_id
        caption = caption.replace("'", "")
        start_time = round(seg_id * frames_per_segment / fps, 1)
        end_time = round(
            min((seg_id + 1) * frames_per_segment / fps, vid_length), 1
        )
        command = f"INSERT INTO tracklets (id, start_time, end_time, caption) \
                  VALUES ({id}, {start_time}, {end_time}, '{caption}')"
        # print(command)
        c.execute(command)
        conn.commit()

    conn.close()
    return sql_path


class OmniVL_VideoDemo(object):
    def __init__(
        self,
        config_path="projects/Omnivl/configs/share/downstream/youcook2_videocaptioning.yaml",
        frames_per_segment=64,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.config_path = config_path
        config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
        config["pretrained"] = "pretrained_models/pretrain_32frm.pth"
        config["input_length"] = frames_per_segment
        config["prompt"] = "In the video, we can see that"
        self.frames_per_segment = frames_per_segment
        self.max_img_size = config["image_size"]
        self.predictor = OmniVLPredictor(config)

    def _frame_from_video(self, video_path):
        # T H W C
        vid_frm_array, fps = load_video_from_path_decord(
            video_path,
            "all",
            num_frm=-1,  # when num_frm=-1, it will load all frames
            height=self.max_img_size,
            width=self.max_img_size,
            return_fps=True,
        )
        vid_length, height, width, _ = vid_frm_array.shape
        num_segments = math.ceil(vid_length / self.frames_per_segment)

        while num_segments <= 4:
            self.frames_per_segment //= 2
            num_segments = math.ceil(vid_length / self.frames_per_segment)

        segments = []
        for i in range(num_segments):
            start = i * self.frames_per_segment
            end = (i + 1) * self.frames_per_segment
            if end > vid_length:
                end = vid_length
            segments.append(vid_frm_array[start:end])

        return segments, vid_length, height, width, fps

    def run_on_video(self, video, question):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        segments, vid_length, _, _, fps = self._frame_from_video(
            video,
        )
        frame_captions = self.predictor(segments)

        sql_path = build_database(
            frame_captions, self.frames_per_segment, vid_length, fps, video
        )
        db = SQLDatabase.from_uri("sqlite:///" + sql_path)
        llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
        if len(question) == 0:
            question = "What is the video about?"

        re = db_chain.run(question)
        return re


class OmniVLPredictor:
    def __init__(self, config):
        selected_device = select_device()
        device = torch.device(selected_device)
        print("Creating OmniVL model on device: ", selected_device)
        
        model = blip_decoder(
            pretrained=config["pretrained"],
            config=config,
            num_frames=config["input_length"],
            temporal_stride=config.get("temporal_stride", 1),
            image_size=config["image_size"],
            vit_grad_ckpt=config["vit_grad_ckpt"],
            vit_ckpt_layer=config["vit_ckpt_layer"],
            prompt=config["prompt"],
            enable_mae=config["enable_mae"],
        )
        self.temporal_stride = config["temporal_stride"]
        model = model.to(device)
        self.omnivl_model = model
        self.norm = VideoNorm()
        self.num_beams = config["num_beams"]
        self.max_length = config["max_length"]
        self.min_length = config["min_length"]
        self.device = device
        

    def caption(self, vid_frm_array, customized_prompt):
        with torch.no_grad():
            outputs = self.omnivl_model.generate(
                vid_frm_array,
                sample=False,
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
                customized_prompt=customized_prompt,
            )
            caption = outputs[0]

        return caption

    def ask(self, vid_frm_array, question):
        with torch.no_grad():
            outputs = self.omnivl_model.generate(
                vid_frm_array,
                sample=False,
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
                customized_prompt=question,
            )
            caption = outputs[0]

        return caption

    def recognize_nouns(self, caption):
        doc = self.ner(caption)
        words = []
        for token in doc:
            if token.pos_ == "NOUN":
                words.append(token.text)
        return words

    def __call__(self, segments, customized_prompt=None):
        caption_list = []
        for segment in segments:
            segment = torch.from_numpy(segment).permute(0, 3, 1, 2).float()
            vid_frm_array = self.norm(segment).unsqueeze(0).to(self.device)
            if vid_frm_array.shape[2] < self.temporal_stride:
                pad_frames = torch.zeros(
                    vid_frm_array.shape[0],
                    vid_frm_array.shape[1],
                    self.temporal_stride - vid_frm_array.shape[2],
                    vid_frm_array.shape[3],
                    vid_frm_array.shape[4],
                ).to(self.device)
                vid_frm_array = torch.cat([vid_frm_array, pad_frames], dim=2)

            caption = self.caption(vid_frm_array, customized_prompt)
            caption_list.append(caption)

        return caption_list


if __name__ == "__main__":
    demo = OmniVL_VideoDemo()
    re = demo.run_on_video(
        "./videos/00f88c4f0a.mp4", "please summarize the video for me."
    )
    print(re)

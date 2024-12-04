import os
import math
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import spacy
import sqlite3
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain


def select_device():
    try:
        os.system("nvidia-smi | grep Default > tmp.txt")
    except:
        return "cpu"

    memorys = []
    with open("tmp.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("|")[2]
            end = line.find("MiB")
            mem = int(line[:end])
            memorys.append(mem)

    gpuid = memorys.index(min(memorys))
    return "cuda:" + str(gpuid)


BLIP2DICT = {
    "FlanT5 XXL": "Salesforce/blip2-flan-t5-xxl",
    "FlanT5 XL COCO": "Salesforce/blip2-flan-t5-xl-coco",
    "OPT6.7B COCO": "Salesforce/blip2-opt-6.7b-coco",
    "OPT2.7B COCO": "Salesforce/blip2-opt-2.7b-coco",
    "FlanT5 XL": "Salesforce/blip2-flan-t5-xl",
    "OPT6.7B": "Salesforce/blip2-opt-6.7b",
    "OPT2.7B": "Salesforce/blip2-opt-2.7b",
}


def build_database(all_cations, height, width, fps, video_path):

    """prompt = "Here is a video: (height: {}, width: {}, length: {}s). ".format(
        height, width, video_length
    )

    for tracklet in all_tracklets:
        prompt += tracklet.generate_description() + "."""

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    sql_path = os.path.join(video_dir, video_name + "blip2.db")
    if os.path.exists(sql_path):
        os.remove(sql_path)
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS tracklets (id INTERGER PRIMARY KEY, time TEXT, caption TEXT)"
    )
    conn.commit()
    for frame_id, caption in enumerate(all_cations):
        time = round(frame_id / fps, 1)
        # caption = add_escape(caption)
        command = f"INSERT INTO tracklets (id, time, caption) \
                  VALUES ({frame_id}, 'at {time} seconds', '{caption}')"
        # print(command)
        c.execute(command)
        conn.commit()

    conn.close()
    return sql_path


class VisualizationBLIP2Demo(object):
    def __init__(self, max_frames=30):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        self.max_frames = max_frames
        self.predictor = BLIP2Predictor()

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
                    # You may need to convert the color.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame)
                    frames.append(frame_pil)

                if count == (end_time * fps):
                    break

            else:
                break

        total_frames = len(frames)
        # print("total frames: ", total_frames, self.max_frames)
        if total_frames > self.max_frames:
            step = total_frames // self.max_frames
            frames = frames[::step]
        else:
            step = 1

        return frames, fps, true_video_length, step

    def run_on_video(self, video, question):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        frames_read, fps, _, step = self._frame_from_video(
            video,
        )
        frame_captions = self.predictor(frames_read)
        height, width = frames_read[0].height, frames_read[0].width
        sql_path = build_database(
            frame_captions, height, width, fps / step, video
        )
        db = SQLDatabase.from_uri("sqlite:///" + sql_path)
        llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
        re = db_chain.run(question)
        return re


class BLIP2Predictor:
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

    def __init__(self, tag="OPT2.7B", bit8=True, device=None):
        self.tag = tag
        self.bit8 = bit8
        dtype = (
            {"load_in_8bit": True}
            if self.bit8
            else {"torch_dtype": torch.float16}
        )

        if device is None:
            selected_device = select_device()
            print("Creating BLIP2 model on device: ", selected_device)
        else:
            selected_device = device

        self.device = torch.device(selected_device)
        self.blip2_processor = Blip2Processor.from_pretrained(
            BLIP2DICT[self.tag]
        )

        mapped_device = (
            int(selected_device[-1])
            if selected_device.startswith("cuda")
            else "cpu"
        )
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2DICT[self.tag], device_map={"": mapped_device}, **dtype
        )
        # python3 -m spacy download en_core_web_lg
        self.ner = spacy.load("en_core_web_lg")

    """
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = "Question: how many cats are there? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    """

    def ask(self, raw_image, question):
        inputs = self.blip2_processor(
            raw_image, question, return_tensors="pt"
        ).to(self.device, torch.float16)
        inputs["max_length"] = 30
        # inputs["min_length"] = 10
        inputs["num_beams"] = 5
        out = self.blip2.generate(**inputs)
        answer = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        return answer

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        caption = self.ask(raw_image, "a photo of")
        caption = caption.replace("\n", " ").strip()  # trim caption
        return caption

    def call_llm(self, prompts):
        prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp["input_ids"].to(self.device)
        attention_mask = prompts_temp["attention_mask"].to(
            self.device, torch.float16
        )

        prompts_embeds = self.blip2.language_model.get_input_embeddings()(
            input_ids
        )

        outputs = self.blip2.language_model.generate(
            inputs_embeds=prompts_embeds, attention_mask=attention_mask
        )

        outputs = self.blip2_processor.decode(
            outputs[0], skip_special_tokens=True
        )

        return outputs

    def recognize_nouns(self, caption):
        doc = self.ner(caption)
        words = []
        for token in doc:
            if token.pos_ == "NOUN":
                words.append(token.text)
        return words

    def __call__(self, frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            frame_captions = []
            for frame in tqdm(frames, desc="Captioning frames"):
                caption = self.caption(frame)
                frame_caption = caption
                frame_captions.append(frame_caption)

        return frame_captions


if __name__ == "__main__":
    demo = VisualizationBLIP2Demo(max_frames=10)
    re = demo.run_on_video(
        "./videos/00f88c4f0a.mp4", "Please summarize the video for me."
    )
    print(re)

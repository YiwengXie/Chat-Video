import os
import cv2
import uuid
import numpy as np
import argparse
from tqdm import tqdm
import inspect
import random
import gradio as gr
from PIL import Image
import re
import torch
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from projects.UNINEXT.uninext.data.datasets.ytvis import (
    YTVIS_CATEGORIES_2019,
)
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog
import multiprocessing as mp

from projects.GroundedSAM2.demo_GroundedSAM2_video import VisualizationDemo
from projects.GroundedSAM2.demo_GroundedSAM2_video_env import VisualizationEnvDemo

os.makedirs("videos", exist_ok=True)

VIDEO_CHATGPT_PREFIX = """Video ChatGPT is designed to assit the users to understanding the video contents. Video ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
As a language model, Video ChatGPT can not directly read videos, but also understanding their contents and recognize the audio and objects within them. Each video will have a file name formed as "videos/xxx.mp4", and when talking about videos, Video ChatGPT is very strict to the file name and will never fabricate nonexistent videos. 
Overall, Video ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
TOOLS:
------
Video ChatGPT  has access to the following tools:"""

VIDEO_CHATGPT_FORMAT_INSTRUCTIONS = """You must use the following format for all responses or your response will be considered incorrect:
```
Question: the input question you must answer
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
If you can't find the answer, say 'I am unable to find the answer.'
You shouldn't take the output of the tools you use as the answer to a human question directly, but only what the question says.
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
Never tell the human the tools that you have access to or you use to answer his question.
"""

VIDEO_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the video file name loyally if it's provided in the last tool observation.
Begin!
Previous conversation history:
{chat_history}
New input: {input}
Since Video ChatGPT is a text language model, Video ChatGPT must use tools to observe videos rather than imagination. You should only answer the question asked by the Human, and not include additional information in you answers.
The thoughts and observations are only visible for Video ChatGPT, Video ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split("\n")
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(" "))
        paragraphs = paragraphs[1:]
    return "\n" + "\n".join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split(".")[0].split("_")
    this_new_uuid = str(uuid.uuid4())[:4]
    most_org_file_name = name_split[-1]
    recent_prev_file_name = name_split[0]
    new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.mp4"
    return os.path.join(head, new_file_name)


def images_to_video(imgs_path, video_path, height, width, fps=None):
    if fps is None:
        fps = 6

    img_array = []
    images = os.listdir(imgs_path)
    images = sorted(images, key=lambda x: int(x.split(".")[0]))

    for i, file_name in enumerate(images):
        file_path = os.path.join(imgs_path, file_name)
        img = cv2.imread(file_path)
        img_array.append(img)

    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def visualize(frames, ori_name, height, width, fps, outputs, dataset):
    video_length = len(frames)
    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs.get("pred_masks", None)
    bboxes = outputs["pred_bboxes"]

    save_path = get_new_image_name(ori_name)
    frame_path = os.path.join(
        os.path.dirname(save_path),
        os.path.basename(ori_name).split(".")[0] + "vis",
    )
    os.makedirs(frame_path, exist_ok=True)

    video_masks = np.zeros(
        (video_length, len(scores), height, width), dtype=np.int32
    )
    video_bboxes = np.zeros((video_length, len(scores), 4), dtype=np.float32)
    video_colors = np.zeros((video_length, len(scores), 3), dtype=np.float32)
    video_labels = np.zeros((video_length, len(scores), 1), dtype=np.int32)
    if dataset == "vis19":
        metadata = MetadataCatalog.get("ytvis_2019_val")
        colormap = YTVIS_CATEGORIES_2019

    elif dataset == "coco":
        metadata = MetadataCatalog.get("coco_2017_val")
        colormap = COCO_CATEGORIES

    if masks is not None:
        for instance_id, (s, l, m, b) in enumerate(
            zip(scores, labels, masks, bboxes)
        ):

            for frame_id, (_mask, _bbox) in enumerate(zip(m, b)):
                if _mask is not None:
                    vis_mask = np.array(_mask).astype(np.int32)
                    vis_color = np.array(metadata.thing_colors[l]).astype(
                        np.int32
                    )
                    video_masks[frame_id][instance_id] = vis_mask
                    video_colors[frame_id][instance_id] = vis_color

                    vis_bbox = np.array(_bbox)
                    video_bboxes[frame_id][instance_id] = vis_bbox
                    video_labels[frame_id][instance_id] = l

        vis_results = []
        for frame_id, (
            video_mask,
            video_bbox,
            video_label,
            video_color,
        ) in enumerate(
            zip(video_masks, video_bboxes, video_labels, video_colors)
        ):
            src = frames[frame_id]

            for ins, insc in zip(video_mask, video_color):
                if ins.sum() > 0:
                    ins = ins[:, :, np.newaxis].repeat(3, axis=2)
                    insc = insc[::-1]
                    color = insc[np.newaxis, np.newaxis, :]
                    ins = (ins * color).astype(src.dtype)
                    src = src + ins * 0.4

            for ins_id, (insb, insl, insc) in enumerate(
                zip(video_bbox, video_label, video_color)
            ):
                if insb.sum() > 0:
                    color_rgb = insc.tolist()
                    color_rgb = [int(c) for c in color_rgb]

                    cv2.rectangle(
                        src,
                        (int(insb[0]), int(insb[1])),
                        (int(insb[2]), int(insb[3])),
                        tuple(color_rgb[::-2]),
                        2,
                    )
                    category = colormap[insl[0]]["name"]

                    text = "{}th ins, {}".format(ins_id, category)
                    cv2.putText(
                        src,
                        text,
                        (int(insb[0]), int(insb[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        tuple(color_rgb[::-2]),
                        2,
                    )

            vis_results.append(src)
            cv2.imwrite(os.path.join(frame_path, str(frame_id) + ".jpg"), src)

    else:
        for instance_id, (s, l, b) in enumerate(zip(scores, labels, bboxes)):

            for frame_id, _bbox in enumerate(b):
                if _bbox is not None:
                    vis_color = np.array(metadata.thing_colors[l]).astype(
                        np.int32
                    )

                    video_colors[frame_id][instance_id] = vis_color

                    vis_bbox = np.array(_bbox)
                    video_bboxes[frame_id][instance_id] = vis_bbox
                    video_labels[frame_id][instance_id] = l

        vis_results = []
        for frame_id, (
            video_bbox,
            video_label,
            video_color,
        ) in enumerate(zip(video_bboxes, video_labels, video_colors)):
            src = frames[frame_id]

            for ins_id, (insb, insl, insc) in enumerate(
                zip(video_bbox, video_label, video_color)
            ):
                if insb.sum() > 0:
                    color_rgb = insc.tolist()
                    color_rgb = [int(c) for c in color_rgb]

                    cv2.rectangle(
                        src,
                        (int(insb[0]), int(insb[1])),
                        (int(insb[2]), int(insb[3])),
                        tuple(color_rgb[::-2]),
                        10,
                    )
                    category = colormap[insl[0]]["name"]

                    text = "{}th ins, {}".format(ins_id, category)
                    cv2.putText(
                        src,
                        text,
                        (int(insb[0]), int(insb[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        tuple(color_rgb[::-2]),
                        2,
                    )

            vis_results.append(src)
            cv2.imwrite(os.path.join(frame_path, str(frame_id) + ".jpg"), src)

    # if fps < 6:
    #     fps = 6

    print(f'\ngenerate video in fps: {fps}\n')
    images_to_video(frame_path, save_path, height, width, fps)
    return save_path


class VideoBasicInformation:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, model_name, dataset):
        mp.set_start_method("spawn", force=True)

    @prompts(
        name="Video Basic Information",
        description="useful when you want to know the height, width, frame-per-seconds, and length of the video. "
        "like: tmp/example.mp4, how long is the video?"
        "The input to this tool should be a comma separated string of two, representing the video path and the question",
    )
    def inference(self, input):
        if len(input.split(",")) == 1:
            filepath = input
            question = ""
            print("Reading the video without question.")
        else:
            # filepath, question = input.split(",")
            sep = input.find(",")
            filepath = input[:sep]
            question = input[sep + 1 :]

        new_path = os.path.join("videos", os.path.basename(filepath))
        if not os.path.exists(new_path):
            os.system("cp {} {}".format(filepath, new_path))
        filepath = new_path

        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        summary = "height of the video is {}, width of the video is {}, frame-per-seconds (FPS) of the video is {}, length of the video is {}.".format(
            height, width, fps, round(length / fps, 1)
        )
        return filepath, summary


class VideoInstanceUnderstanding:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, model_name, dataset):
        mp.set_start_method("spawn", force=True)
        self.model = VisualizationDemo(dataset)
        self.dataset = dataset

    @prompts(
        name="VideoInstanceUnderstanding",
        description="useful when you want to summarize the video, or when you want to know the events or activities in this video, or the number, appearance, motion, spatial location, and even trajectory of the objects in the video."
        "The input to this tool should be a comma separated string of two, representing the video path and the question. The video name ends with .mp4, and the text after .mp4 is the question.",
    )
    def inference(self, input):
        # print("input is :", input)

        if len(input.split(",")) == 1:
            filepath = input
            question = ""
            print("Reading the video without question.")
        else:
            # filepath, question = input.split(",")
            sep = input.find(",")
            filepath = input[:sep]
            question = input[sep + 1 :]
            print("Reading the video with question: {}".format(question))

        new_path = os.path.join("videos", os.path.basename(filepath))
        if not os.path.exists(new_path):
            os.system("cp {} {}".format(filepath, new_path))
        filepath = new_path

        re, model_outputs, video_info = self.model.run_on_video(
            filepath, question
        )

        if len(question) == 0:
            print("Visualization without question")
            if model_outputs is None:
                print("Already tracked.")
                vid_name = os.path.basename(filepath).split(".")[0]
                vids = os.listdir(os.path.dirname(filepath))
                vis_path = None
                for v in vids:
                    if vid_name in v and "mp4" in v and "update" in v:
                        vis_path = os.path.join(os.path.dirname(filepath), v)

                vis_path = filepath if vis_path is None else vis_path
            else:
                # print('fps:', video_info["fps"], '        step:',video_info['step'])
                vis_path = visualize(
                    video_info["frames"],
                    filepath,
                    video_info["height"],
                    video_info["width"],
                    video_info["fps"] / video_info['step'],
                    model_outputs,
                    self.dataset,
                )
        else:
            vis_path = filepath

        return vis_path, re


class VideoEnvUnderstanding:
    def __init__(self, model_name, dataset):
        mp.set_start_method("spawn", force=True)
        self.model = VisualizationEnvDemo()

    @prompts(
        name="VideoEnvUnderstanding",
        description="useful when you want to know the environment and sounds in the video. The input to this tool should include a video_path and a quesiton, separated by a comma.",
    )
    def inference(self, input):
        if len(input.split(",")) == 1:
            filepath = input
            question = ""
            print("Reading the video without question.")
        else:
            sep = input.find(",")
            filepath = input[:sep]
            question = input[sep + 1 :]

        new_path = os.path.join("videos", os.path.basename(filepath))
        if not os.path.exists(new_path):
            os.system("cp {} {}".format(filepath, new_path))
        filepath = new_path
        summary = self.model.run_on_video(filepath, question)

        return filepath, summary


class ConversationBot:
    def __init__(self, model_name, dataset, methods):
        self.llm = OpenAI(temperature=0,
                          openai_api_key=os.getenv("OPENAI_API_KEY"),
                          model="gpt-3.5-turbo-instruct"
                          )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )

        self.models = {}
        # Load Basic Foundation Models
        for method in methods:
            self.models[method] = globals()[method](
                model_name, dataset
            )

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k
                    for k in inspect.signature(
                        module.__init__
                    ).parameters.keys()
                    if k != "self"
                }
                loaded_names = set(
                    [type(e).__name__ for e in self.models.values()]
                )
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{
                            name: self.models[name]
                            for name in template_required_names
                        }
                    )
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(
                            name=func.name,
                            description=func.description,
                            func=func,
                        )
                    )
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": VIDEO_CHATGPT_PREFIX,
                "format_instructions": VIDEO_CHATGPT_FORMAT_INSTRUCTIONS,
                "suffix": VIDEO_CHATGPT_SUFFIX,
            },
        )

    def set_frames(self, frames):
        for name, model in self.models.items():
            if (
                "VideoInstanceUnderstanding" in name
                or "VideoEnvUnderstanding" in name
            ):
                model.model.set_maxframes(frames)

    def set_fps(self, fps):
        for name, model in self.models.items():
            if ("VideoInstanceUnderstanding" in name
                or "VideoEnvUnderstanding" in name):
                model.model.set_fps(fps)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=500
        )

        try:
            res = self.agent({"input": text.strip()})
        except ValueError as e:
            response = str(e)
            res = {"output": response}

        res["output"] = res["output"].replace("\\", "/")
        
        new_input_index = res["output"].find('New input')
        if new_input_index != -1:
            res["output"] = res["output"][:new_input_index].strip()
        else:
            res["output"] = res["output"]

        response = re.sub(
            "(videos/\S*mp4)",
            lambda m: f"![](/file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )
        state = state + [(text, response)]

        return state, state

    def run_image(self, image_filename, state, txt):
        vis_path, description = self.models["VideoBasicInformation"].inference(
            image_filename
        )
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say "Received". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [
            (f"![](/file={image_filename})*{image_filename}*", AI_prompt)
        ]
        print(
            f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: \n{state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )

        return vis_path, state, state, f"{txt} {image_filename} "

    def run_track(self, image_filename, state, txt):
        vis_path, description = self.models[
            "VideoInstanceUnderstanding"
        ].inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say "Received". \n'
        AI_prompt = "Tracked.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [
            (f"![](/file={image_filename})*{image_filename}*", AI_prompt)
        ]
        print(
            f"\nProcessed run_track, Input image: {image_filename}\nCurrent state: \n{state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return vis_path, state, state, f"{txt}"

def get_config():
    parser = argparse.ArgumentParser(description='Chat-Video', add_help=False)
    parser.add_argument('--model_name', type=str, default='r50', choices=['r50', 'conv-large', 'vit-huge'])
    parser.add_argument('--dataset', type=str, default='coco', choices=['vis19', 'coco'])
    parser.add_argument('--load', type=str, default='VideoBasicInformation,VideoInstanceUnderstanding,VideoEnvUnderstanding')
    
    return parser


if __name__ == "__main__":
    # set_seed(666666)
    gr.close_all()
    parser = get_config()
    args = parser.parse_args()
    load_methods = args.load.split(",")
    bot = ConversationBot(
        args.model_name, args.dataset, load_methods
    )

    with gr.Blocks(
        css="#chatbot .overflow-y-auto{height:500px}",
    ) as demo:
        gr.Markdown(
            "# ChatVideo: Make the video understanding easier for everyone",
        )
        gr.Markdown(
            "### You can **upload** a video, **track** the objects in it, and ask me any question about the video."
        )

        with gr.Row():
            with gr.Column(scale=0.5):
                vid1 = gr.Video(source="upload", interactivate=True).style(
                    height=300
                )
                slider1 = gr.Slider(
                    1,
                    60,
                    value=30,
                    interactive=True,
                    info="set fps",
                    step=1,
                    label="Fps to be set",
                )
                slider2 = gr.Slider(
                    1,
                    2000,
                    value=200,
                    interactive=True,
                    info="decrease this value for faster response and increase it to get more accurate answers.",
                    step=1,
                    label="Maximum frames to be sampled",
                )
                btn = gr.Button("Track")
                vid2 = gr.Video(
                    label="Tracking Results",
                    show_label=True,
                    show_progress=True,
                ).style(height=300)

            with gr.Column(scale=0.5, min_width=0):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                ).style(container=False)

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="A chat bot for video understanding",
                ).style(height=650)
                clear = gr.Button("Clear")

        slider1.release(bot.set_fps, [slider1], None)
        slider2.release(bot.set_frames, [slider2], None)

        state = gr.State([])
        vid1.upload(
            bot.run_image, [vid1, state, txt], [vid1, chatbot, state, txt]
        )

        btn.click(
            bot.run_track, [vid1, state, txt], [vid2, chatbot, state, txt]
        )

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)

        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        close_btn = gr.Button("Close")
        def shutdown():
            return gr.Chatbot.update(value=[("System", "Server is shutting down, please close the client window.")])
        close_btn.click(shutdown, None, chatbot)
        close_btn.click(lambda: os._exit(0)) 

        demo.launch(server_name="0.0.0.0", server_port=8090, share=True)
import os
import librosa
import torch
import copy
import torchaudio.transforms as transforms
from transformers import (
    pipeline,
    ASTFeatureExtractor,
    AutoModelForAudioClassification,
)
from moviepy.editor import AudioFileClip
from deepmultilingualpunctuation import PunctuationModel
from IPython.display import Audio

asr_model = pipeline(
    "automatic-speech-recognition",
    "facebook/wav2vec2-base-960h",
    device=0 if torch.cuda.is_available() else -1,
)

emotion_model = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=0 if torch.cuda.is_available() else -1,
)


def generate_caption(category, speech, emotion):
    prompt = "It seems the sound in the video is of a " + category + ". "
    if speech is not None:
        prompt += (
            "And the speaker(s) is "
            + emotion
            + '. The speaker(s) is saying "'
            + speech
            + '".'
        )

    return prompt


class VisualizationAudioDemo(object):
    def __init__(
        self,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ASTFeatureExtractor()
        self.ast_model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(device)
        category_mappings = self.ast_model.config.id2label
        human_sounds = []
        for k, v in category_mappings.items():
            if 0 <= k <= 71:
                human_sounds.append(v)
        self.human_sounds = human_sounds
        self.rpunct_model = PunctuationModel()

        self.emotion_mapping = {
            "neu": "neural",
            "hap": "happy",
            "sad": "sad",
            "ang": "angry",
        }

        self.device = device

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

    def run_on_video(self, video_path):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        try:
            my_audio_clip = AudioFileClip(video_path)
            audio_path = os.path.join(
                os.path.dirname(video_path), "extract_audio.wav"
            )
            my_audio_clip.write_audiofile(audio_path)
            my_audio_clip.close()
        except Exception as e:
            print("No audio in this video or fail to extract the audio: ", e)
            return None

        category = self.classification_ast(audio_path)

        if category in self.human_sounds:
            with torch.no_grad():
                text = asr_model(audio_path)["text"]
                text = self.rpunct_model.restore_punctuation(text.lower())
                emotion = emotion_model(audio_path, top_k=1)[0]["label"]
            emotion = self.emotion_mapping[emotion]
        else:
            text = None
            emotion = None

        audio_description = generate_caption(category, text, emotion)
        # print(audio_description)
        return audio_description

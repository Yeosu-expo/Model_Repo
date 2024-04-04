import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2",device=0)

    def infer(self, audio_url):
        pipeline_output = self.generator(audio_url)
        generated_txt = pipeline_output["text"]
        return generated_txt

    def finalize(self):
        self.pipe = None

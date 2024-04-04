import app
import json
import os
import base64
import triton_python_backend_utils as pb_utils
import numpy as np
import time

inferless_model = app.InferlessPythonModel()

class TritonPythonModel:
    def initialize(self, args):
        inferless_model.initialize()
        self.audio_file_dir = "./"

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "audio_url")
            audio_base64 = input_tensor.as_numpy()[0]
            audio_bytes = base64.b64decode(audio_base64)
            
            audio_file_name = self.save_audio_file(audio_bytes)
            
            output = inferless_model.infer(self.audio_file_dir + audio_file_name)
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "transcribed_text",
                        np.array([output.encode()]),
                    )
                ]
            )
            responses.append(inference_response)
        return responses
    
    def save_audio_file(self, audio_bytes):
        unique_id = int(time.time() * 1000)
        file_name = f"audio_{unique_id}.flac" # 고유 ID 생성 로직 필요
        file_path = os.path.join(self.audio_file_dir, file_name)
        with open(file_path, 'wb') as audio_file:
            audio_file.write(audio_bytes)
        print(file_name)
        return file_name

    def finalize(self, args):
        inferless_model.finalize()


class KimDeaMin:
	def __init__(self):
		self.name = "바보"
	
	def __str__(self):
		return self.name



class AnJunCheol:
	def __init__(self):
		self.name = "지니어스"
	
	def __str__(self):
		return self.name

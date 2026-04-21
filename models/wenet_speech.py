import os

from models.base_model import BaseASRModel
from models.registry import register_model

DEFAULT_MODEL_PATH = "/home/qf/lyn/asr_benchmark/init_model/wenetspeech"
MODEL_PATH_ENV = "ASR_MODEL_WENET_SPEECH_PATH"


@register_model("wenet_speech")
class WenetSpeechModel(BaseASRModel):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(model_name="wenet_speech", device=device)

    def load_model(self):
        try:
            import wenet
        except Exception as e:
            raise RuntimeError(
                "wenet_speech requires 'wenet' package. Please install wenet first."
            ) from e

        model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"wenet_speech model path not found: {model_path}. "
                f"Override with {MODEL_PATH_ENV}."
            )

        try:
            self.model = wenet.load_model(model_path)
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize wenet_speech from '{model_path}': {str(e)}"
            ) from e

    def infer(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        raise RuntimeError(f"Unexpected wenet_speech output: {result!r}")


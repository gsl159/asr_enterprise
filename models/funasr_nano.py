import os

from funasr import AutoModel

from models.base_model import BaseASRModel
from models.registry import register_model

DEFAULT_MODEL_PATH = "/home/qf/lyn/asr_benchmark/init_model/Fun-ASR-Nano-2512"
MODEL_PATH_ENV = "ASR_MODEL_FUNASR_NANO_PATH"


@register_model("funasr_nano")
class FunasrNano(BaseASRModel):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(model_name="funasr_nano", device=device)

    def load_model(self):
        model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"funasr_nano model path not found: {model_path}. "
                f"Override with {MODEL_PATH_ENV}."
            )

        try:
            self.model = AutoModel(model=model_path, device=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize funasr_nano from '{model_path}': {str(e)}"
            ) from e

    def infer(self, audio_path: str) -> str:
        result = self.model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            language="zh",
            itn=False,
        )
        if isinstance(result, list) and result and isinstance(result[0], dict):
            text = result[0].get("text")
            if isinstance(text, str):
                return text
        raise RuntimeError(f"Unexpected funasr_nano output: {result!r}")


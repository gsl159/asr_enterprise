import os

from funasr import AutoModel

from models.base_model import BaseASRModel
from models.registry import register_model

DEFAULT_MODEL_PATH = "/data0/lyn/model/paraformer/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_PATH_ENV = "ASR_MODEL_PARAFORMER_QF_PATH"


@register_model("paraformer_qf")
class ParaformerQfModel(BaseASRModel):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(model_name="paraformer_qf", device=device)

    def load_model(self):
        model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        if os.path.isabs(model_path) and not os.path.exists(model_path):
            raise RuntimeError(
                f"paraformer_qf model path not found: {model_path}. "
                f"Override with {MODEL_PATH_ENV}."
            )

        try:
            self.model = AutoModel(model=model_path, device=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize paraformer_qf from '{model_path}': {str(e)}"
            ) from e

    def infer(self, audio_path: str) -> str:
        result = self.model.generate(input=audio_path, batch_size_s=300)
        if isinstance(result, list) and result and isinstance(result[0], dict):
            text = result[0].get("text")
            if isinstance(text, str):
                return text
        raise RuntimeError(f"Unexpected paraformer_qf output: {result!r}")


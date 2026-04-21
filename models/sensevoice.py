import os

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from models.base_model import BaseASRModel
from models.registry import register_model

DEFAULT_MODEL_PATH = "iic/SenseVoiceSmall"
MODEL_PATH_ENV = "ASR_MODEL_SENSEVOICE_PATH"


def _ensure_modelscope():
    try:
        import modelscope  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "sensevoice requires 'modelscope'. Please run: pip install modelscope"
        ) from e


@register_model("sensevoice")
class SenseVoiceModel(BaseASRModel):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(model_name="sensevoice", device=device)

    def load_model(self):
        _ensure_modelscope()
        model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        if os.path.isabs(model_path) and not os.path.exists(model_path):
            raise RuntimeError(
                f"sensevoice model path not found: {model_path}. "
                f"Override with {MODEL_PATH_ENV}."
            )

        try:
            self.model = AutoModel(
                model=model_path,
                trust_remote_code=True,
                device=self.device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize sensevoice from '{model_path}': {str(e)}"
            ) from e

    def infer(self, audio_path: str) -> str:
        result = self.model.generate(
            input=audio_path,
            cache={},
            language="zh",
            use_itn=True,
            batch_size=64,
        )

        if isinstance(result, list) and result and isinstance(result[0], dict):
            text = result[0].get("text")
            if isinstance(text, str):
                return rich_transcription_postprocess(text)

        raise RuntimeError(f"Unexpected sensevoice output: {result!r}")


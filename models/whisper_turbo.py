import whisper
from models.base_model import BaseASRModel
from models.registry import register_model


@register_model("whisper_turbo")
class WhisperTurboModel(BaseASRModel):

    def __init__(self, device="cuda:0"):
        super().__init__("whisper_turbo", device)

    def load_model(self):
        self.model = whisper.load_model("turbo")
        if "cuda" in self.device:
            self.model = self.model.to(self.device)

    def infer(self, audio_path: str) -> str:
        result = self.model.transcribe(
            audio_path,
            language="zh"
        )
        return result["text"]

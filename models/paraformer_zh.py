from funasr import AutoModel

from models.base_model import BaseASRModel
from models.registry import register_model


@register_model("paraformer_zh")
class ParaformerZhModel(BaseASRModel):

    def __init__(self, device="cuda:0"):
        super().__init__("paraformer_zh", device)

    def load_model(self):
        try:
            self.model = AutoModel(
                model="iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize paraformer_zh: {str(e)}"
            ) from e

    def infer(self, audio_path: str) -> str:
        res = self.model.generate(input=audio_path)
        return res[0]["text"]

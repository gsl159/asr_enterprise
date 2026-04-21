import yaml


class Config:

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.data.get(key, default)

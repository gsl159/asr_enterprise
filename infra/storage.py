import os


class Storage:

    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

import os
import json


class SafeWriter:

    @staticmethod
    def write_json(path, data):

        tmp_path = path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

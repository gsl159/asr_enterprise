import torch
import threading


class GPUManager:

    def __init__(self, auto_detect=False, devices=None):

        if auto_detect:
            count = torch.cuda.device_count()
            if count == 0:
                self.devices = ["cpu"]
            else:
                self.devices = [f"cuda:{i}" for i in range(count)]
        else:
            self.devices = devices or ["cpu"]

        self.lock = threading.Lock()
        self.index = 0

    def acquire(self):
        with self.lock:
            device = self.devices[self.index]
            self.index = (self.index + 1) % len(self.devices)
            return device

class ASRException(Exception):
    pass


class ModelLoadException(ASRException):
    pass


class InferenceException(ASRException):
    pass


class GPUUnavailableException(ASRException):
    pass

import os
import numpy as np

class EmptyAIInferencer:
    def __init__(self):
        pass

    def put(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        pass

    def release(self):
        pass


class AIInferencer:
    def __init__(self, model_path:str|None, cores:tuple[int]=(0,), pool_mode:bool=True):
        """
        args:
            model_path: 模型文件路径 str: '/path/to/model'
            会根据文件后缀名自动识别模型类型, 目前支持rknn, onnx, tflite

            cores: 指定使用哪些核心进行推理 tuple: (0,) or any like (0, 1, 2)
            默认使用第一个核心. 若使用多个核心, 则开启多个线程

            pool_mode: 线程池推理模式 bool: True or False
            若为True, 则使用线程池进行推理, 否则使用单线程推理器
        """
        self.model_path = model_path
        if isinstance(cores, int):
            cores = tuple([cores])
        self.cores = cores
        self.pool_mode = pool_mode

        self.inferfacer = EmptyAIInferencer()
        self.inferfacer_init()

    @staticmethod
    def identify_model_type(model_path):
        """
        根据模型文件的后缀来识别模型种类
        """
        ext:str = os.path.splitext(model_path)[-1]
        ext = ext.lower()  # 将后缀转换为小写，以便于比较

        model_type = 'Unknown'

        if ext == '.rknn':
            model_type = 'rknn'
        elif ext == '.onnx':
            model_type = 'onnx'
        elif ext == '.bin':
            model_type = 'qnn'

        return model_type
    
    def inferfacer_init(self):
        self.model_type = self.identify_model_type(self.model_path)

        if self.model_type == 'rknn':
            if self.pool_mode is True:
                from .rknn_executor import RknnThreadPool
                self.inferfacer = RknnThreadPool(self.model_path, self.cores)

            else:
                from .rknn_executor import RknnExecutor
                self.inferfacer = RknnExecutor(self.model_path, self.cores)

        elif self.model_type == 'qnn':
            if self.pool_mode is True:
                from .qnn_excutor import QnnProcessPool
                self.inferfacer = QnnProcessPool(self.model_path, self.cores)

            else:
                from .qnn_excutor import QnnExecutor
                self.inferfacer = QnnExecutor(self.model_path)
    
    def release(self) -> bool:
        ret = self.inferfacer.release()
        return ret
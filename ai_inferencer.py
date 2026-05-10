import os
import sys
from pathlib import Path
import numpy as np


CURRENT_DIR = Path(__file__).parent.resolve()

class temporary_sys_path:
    def __init__(self, new_path: str):
        self.new_path = str(new_path)
        
    def __enter__(self):
        sys.path.insert(0, self.new_path)
        return self
        
    def __exit__(self, etype, value, traceback):
        # 安全移除：避免原本就在 sys.path 里导致误删
        if self.new_path in sys.path:
            sys.path.remove(self.new_path)



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
        self.model_path = str(model_path)
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
        elif ext == '.bin':
            model_type = 'qnn'
        elif ext == '.onnx':
            model_type = 'onnx'


        return model_type
    
    def inferfacer_init(self):
        self.model_type = self.identify_model_type(self.model_path)

        if self.model_type == 'rknn':
            with temporary_sys_path(CURRENT_DIR):
                import rknn_inferencer as rknn_infer

            if self.pool_mode is True:
                self.inferfacer = rknn_infer.RknnThreadPool(self.model_path, self.cores)
            else:
                self.inferfacer = rknn_infer.RknnExecutor(self.model_path, self.cores)

        elif self.model_type == 'qnn':
            with temporary_sys_path(CURRENT_DIR):
                import qnn_inferencer as qnn_infer

            if self.pool_mode is True:
                self.inferfacer = qnn_infer.QnnProcessPool(self.model_path, self.cores)
            else:
                self.inferfacer = qnn_infer.QnnExecutor(self.model_path)

        elif self.model_type == 'onnx':
            with temporary_sys_path(CURRENT_DIR):
                import onnx_inferencer as onnx_infer

            self.inferfacer = onnx_infer.OnnxExecutor(self.model_path)
    
    def release(self) -> bool:
        ret = self.inferfacer.release()
        return ret
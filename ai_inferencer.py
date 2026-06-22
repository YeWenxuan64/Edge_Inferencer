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
    def __init__(self, model_path:str|None, cores:tuple[int]=(0,), mult_task:bool=False):
        """
        args:
            model_path: 模型文件路径 str: '/path/to/model'
            会根据文件后缀名自动识别模型类型, 目前支持rknn, onnx, tflite

            cores: 指定使用哪些核心进行推理 tuple: (0,) or any like (0, 1, 2)
            默认使用第一个核心. 若使用多个核心, 则开启多个线程

            mult_task: 线程池推理模式 bool: True or False
            若为True, 则使用线程池进行推理, 否则使用单线程推理器
        """
        
        self.model_path = str(model_path)

        if isinstance(cores, int):
            cores = tuple([cores])
        self.cores = cores
        self.mult_task = mult_task

        self.inferfacer = EmptyAIInferencer()
        self.inferfacer_init()

    @staticmethod
    def identify_model_type(model_path:str|None):
        """
        根据模型文件的后缀来识别模型种类
        """
        if model_path:
            p = Path(model_path)
            if not p.exists():
                print(f"Model file not found: {model_path}")
            ext = p.suffix.lower()
        else:
            ext = ''

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

            if self.mult_task:
                self.inferfacer = rknn_infer.RknnThreadPool(self.model_path, self.cores)
            else:
                self.inferfacer = rknn_infer.RknnExecutor(self.model_path, self.cores)

        elif self.model_type == 'qnn':
            with temporary_sys_path(CURRENT_DIR):
                import qnn_inferencer as qnn_infer

            if self.mult_task:
                self.inferfacer = qnn_infer.QnnProcessPool(self.model_path, self.cores)
                #self.inferfacer = qnn_infer.QnnThreadPool(self.model_path, self.cores)
            else:
                #self.inferfacer = qnn_infer.QnnExecutor(self.model_path)
                self.inferfacer = qnn_infer.QnnExecutor2(self.model_path)
                #self.inferfacer = qnn_infer.QnnExecutor3(self.model_path)

        elif self.model_type == 'onnx':
            with temporary_sys_path(CURRENT_DIR):
                import onnx_inferencer as onnx_infer

            self.inferfacer = onnx_infer.OnnxExecutor(self.model_path)
    
    def release(self) -> bool:
        ret = self.inferfacer.release()
        return ret
    



def timeit(func=None, *, measure_cycle_time:bool=False):
    """Decorator that measures and reports execution time / FPS of the wrapped function.

    Supports two usage forms:
        @timeit
        @timeit(measure_cycle_time=True)

    Args:
        func: The function to wrap (auto-filled when used as @timeit without parentheses).
        measure_cycle_time: If False (default), measures single-call duration (end - start).
                            If True, measures cycle time between consecutive calls (start - last_start),
                            which includes any idle/wait time between invocations.

    Output (printed every ~1s):
        Default:    func_name: X.XXX ms, fps: YYY.YYY
        Cycle mode: func_name (per cycle): X.XXX ms, fps: YYY.YYY
    """
    import time
    from functools import wraps
    from collections import deque

    time_length = 30  # rolling window size for averaging
    time_list = deque(maxlen=time_length)
    last_print_time = time.perf_counter()  # tracks when we last printed stats

    def decorator(func):
        last_start_time = time.perf_counter()  # for cycle-time measurement

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_print_time, last_start_time

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end = time.perf_counter()

                # Measure: single-call duration or cycle interval
                if measure_cycle_time:
                    elapsed = start - last_start_time
                    last_start_time = start
                else:
                    elapsed = end - start

                time_list.append(elapsed)

                # Print rolling average every ~1s
                if end - last_print_time >= 1.0:
                    last_print_time = end
                    mean_time = sum(time_list) / time_length

                    if measure_cycle_time:
                        print(f"{func.__name__} (per cycle): {mean_time*1000:.3f} ms, fps: {1.0/(mean_time + 1e-7):.3f}")
                    else:
                        print(f"{func.__name__}: {mean_time*1000:.3f} ms, fps: {1.0/(mean_time + 1e-7):.3f}")

            return result
        return wrapper

    # Support both @timeit and @timeit(measure_cycle_time=True)
    if func is None:
        return decorator
    return decorator(func)
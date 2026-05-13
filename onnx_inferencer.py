
import numpy as np
import onnxruntime as ort
from onnxruntime import NodeArg


class OnnxExecutor():
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.session = None
        self.set_providers()

        self.input_names = []
        self.output_names = []
        self.float_inputs = False

        self.last_outputs:list[np.ndarray]|None = None

    def set_providers(self, providers:list[str]=['CPUExecutionProvider']):
        self.providers = providers

    def init_onnx(self):
        # 初始化 ONNX Runtime Session
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        input_details:list[NodeArg] = self.session.get_inputs()

        self.input_names = [inp.name for inp in input_details]
        self.output_names = [out.name for out in self.session.get_outputs()]

        if "float" in input_details[0].type:
            self.float_inputs = True
        
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]:
        if self.session is None:
            self.init_onnx()

        if input_format == 'nhwc':
            input_data = [np.transpose(tensor, (0, 3, 1, 2)) for tensor in input_data]
        elif input_format == 'nchw':
            pass

        if self.float_inputs:
            input_data = [tensor.astype(np.float32) for tensor in input_data]

        input_feed = {} # 构建 feed_dict

        for i, input_name in enumerate(self.input_names):
            input_feed[input_name] = input_data[i]
            
        outputs = self.session.run(None, input_feed) # 执行推理
        self.last_outputs = outputs

        return outputs
    
    def get(self, block:bool=True) -> list[np.ndarray]:
        ret = self.last_outputs
        self.last_outputs = None

        return ret

    def release(self):
        if self.session is not None:
            del self.session
            self.session = None

            self.input_names.clear()
            self.output_names.clear()

        print("ONNX Executor released")
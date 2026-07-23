import os
import platform
import numpy as np
import onnxruntime as ort
from onnxruntime import NodeArg, SessionOptions, ExecutionMode, RunOptions
from concurrent.futures import ThreadPoolExecutor, Future


def set_onnx_session_options(task_num:int=1) -> SessionOptions|None:
    sess_options = None
    architecture = platform.machine().lower()

    if platform.machine().lower() in ["amd64", "x86", "x86_64", "i386"]:
        sess_options = SessionOptions()


        logic_cpu_counts = os.cpu_count()
        if logic_cpu_counts is not None:
            cpu_count = logic_cpu_counts // 2
        else:
            cpu_count = 4

        thread_per_task = max(1, cpu_count // task_num)

        if task_num == 1:
            print(f"ONNX Executor initialized in {architecture} with {cpu_count} threads")
        else:
            print(f"ONNX Threadpool initialized with {task_num} tasks, {thread_per_task} thread_per_task")
        
        sess_options.intra_op_num_threads = thread_per_task  # 单个算子内并行线程数
        sess_options.inter_op_num_threads = 1   # 算子间并行线程数
        sess_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL  # 顺序执行

    return sess_options

def get_onnxruntime_metadata(session:ort.InferenceSession) -> tuple[list[str], list[str], bool]:
    input_details:list[NodeArg] = session.get_inputs()

    float_inputs = False
    input_names:list[str] = [inp.name for inp in input_details]
    output_names:list[str]  = [out.name for out in session.get_outputs()]

    if "float" in input_details[0].type:
        float_inputs = True

    print(f"ONNX input_names: {input_names}, output_names: {output_names}, is float_inputs: {float_inputs}")
    return input_names, output_names, float_inputs


class OnnxExecutor():
    def __init__(self, model_path:str):
        """
        Initialize the ONNX executor.

        Args:
            model_path: Path to the ONNX model file.
        """
        self.model_path = model_path
        self.session = None
        self.set_providers()

        self.input_names:list[str] = []
        self.output_names:list[str] = []
        self.float_inputs = False

    def set_providers(self, providers:list[str]=['CPUExecutionProvider']):
        """
        Set the ONNX Runtime execution providers.

        Args:
            providers: List of execution provider names.
                Defaults to ['CPUExecutionProvider'].
        """
        self.providers = providers

    def init_onnx(self):
        """
        Initialize the ONNX Runtime inference session.

        Creates an InferenceSession from the model path and retrieves
        input/output metadata. Must be called before the first inference.
        """
        sess_options = set_onnx_session_options()

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)
        self.input_names, self.output_names, self.float_inputs = get_onnxruntime_metadata(self.session)
        
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]:
        """
        Inference with blocking.

        Args:
            input_data: a list of ndarrays
            input_format: 'nhwc' or 'nchw'
                - input_tensor should [n, h, w, c] or [n, c, h, w]. 
                - recommended [n, c, h, w], and model's input_format should be [n, c, h, w].

        Returns:
            a list of ndarrays
        """

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
        return outputs
    
    def release(self):
        """
        Release the ONNX Runtime session and clear all resources.
        """
        if self.session is not None:
            del self.session
            self.session = None

            self.input_names.clear()
            self.output_names.clear()

            print("ONNX Executor released")


class OnnxThreadPool():
    def __init__(self, model_path:str, task_num_or_cores:int|tuple[int]=2):
        """
        Initialize the ONNX thread pool with multiple inference sessions.

        Args:
            model_path: Path to the ONNX model file.
            task_num_or_cores: Number of concurrent inference tasks / sessions.
                               If an integer, that many sessions are created.
                               If a tuple (e.g., core indices), its length is
                               used as the session count.
                               Defaults to 2.
        """
        self.model_path = model_path
        if isinstance(task_num_or_cores, int):
            self.task_num = task_num_or_cores
        else:
            self.task_num = len(task_num_or_cores)
        
        self.set_providers()

        self.thread_pool:ThreadPoolExecutor|None = None
        self.session_list:list[ort.InferenceSession] = []
        self.queue_list:list[Future] = []

        self.input_names:list[str] = []
        self.output_names:list[str] = []
        self.float_inputs:bool = False

        self.frame_index:int = 0

    def set_providers(self, providers:list[str]=['CPUExecutionProvider']):
        """
        Set the ONNX Runtime execution providers for all sessions.

        Args:
            providers: List of execution provider names.
                       Defaults to ['CPUExecutionProvider'].
        """
        self.providers = providers

    def init_onnxruntime_threadpool(self) -> tuple[ThreadPoolExecutor, list[ort.InferenceSession]]:
        """
        Create one InferenceSession per worker thread and a shared thread pool.

        Returns:
            A tuple (thread_pool, session_list):
                - thread_pool: ThreadPoolExecutor with task_nums workers.
                - session_list: List of InferenceSession objects, one per worker.
        """
        sess_options = set_onnx_session_options(self.task_num)

        session_list:list[ort.InferenceSession] = []
        for i in range(self.task_num):
            session = ort.InferenceSession(self.model_path, sess_options, providers=['CPUExecutionProvider'])
            session_list.append(session)

            if i == 0: # 从第一个 session 获取模型元信息
                self.input_names, self.output_names, self.float_inputs = get_onnxruntime_metadata(session)

        thread_pool = ThreadPoolExecutor(max_workers=self.task_num, thread_name_prefix='onnx_task_pool')
        return thread_pool, session_list

    def onnx_inference(self, session:ort.InferenceSession, input_feed:dict[str, np.ndarray]) -> list[np.ndarray]|None:
        """
        Run synchronous ONNX inference inside a worker thread.

        Args:
            session: The InferenceSession assigned to this worker.
            input_feed: Dict mapping input names to tensors.

        Returns:
            List of output ndarrays, or None on failure.
        """
        outputs = session.run(self.output_names, input_feed)
        return outputs

    def queue_put(self, input_data:list[np.ndarray], allow_drop:bool=True) -> None:
        """
        Submit an inference task to the thread pool with round-robin dispatch.

        Args:
            input_data: Pre-processed input tensor list.
            allow_drop: If True, skip submission when the queue is full and
                        the oldest task has not yet completed.
        """
        if allow_drop and len(self.queue_list) >= self.task_num:
            if not self.queue_list[0].done():
                return

        idx = self.frame_index

        input_feed = {}
        for i, name in enumerate(self.input_names):
            input_feed[name] = input_data[i]

        future = self.thread_pool.submit(self.onnx_inference, self.session_list[idx], input_feed)
        self.queue_list.append(future)

        self.frame_index = (self.frame_index + 1) % self.task_num

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> None:
        """
        Submit an inference task (non-blocking).

        On first call, lazily initializes the thread pool and all sessions.
        All sessions are initially primed with the same frame for consistent
        behaviour with the RKNN backend.

        Args:
            input_data: List of input tensors.
            input_format: 'nhwc' or 'nchw'. 'nhwc' tensors are transposed
                          to 'nchw' automatically.
        """

        if input_format == 'nhwc':
            input_data = [np.transpose(tensor, (0, 3, 1, 2)) for tensor in input_data]
        elif input_format == 'nchw':
            pass

        if self.float_inputs:
            input_data = [tensor.astype(np.float32) for tensor in input_data]

        if self.thread_pool:
            self.queue_put(input_data)

        else:
            self.thread_pool, self.session_list = self.init_onnxruntime_threadpool()

            if self.float_inputs:
                input_data = [tensor.astype(np.float32) for tensor in input_data]

            for i in range(self.task_num):
                self.queue_put(input_data, allow_drop=False)
                print(f'ONNX task pool: session {i} submitted')

            if self.task_num == 1:
                self.queue_put(input_data, allow_drop=False)

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        """
        Retrieve an inference result from the queue.

        Args:
            block: If True, block until a result is available.
                   If False, return None immediately if no result is ready.

        Returns:
            The inference output list, or None if no result is available.
        """
        if not self.queue_list:
            return None

        future:Future = self.queue_list.pop(0)

        if block is False and future.done() is False:
            self.queue_list.insert(0, future)
            return None

        try:
            result = future.result()
        except Exception as e:
            print(f"ONNX ThreadPool get error: {e}")
            result = None

        return result

    def release(self) -> bool:
        """
        Release the thread pool and all session resources.

        Returns:
            True if resources were released, False if already released.
        """
        ret = False
        
        if self.thread_pool is not None:
            for future in self.queue_list:
                future.cancel()
            self.queue_list.clear()

            self.thread_pool.shutdown(wait=True, cancel_futures=True)
            
            for i, session in enumerate(self.session_list):
                del session
                print(f'ONNX session {i} released')
            self.session_list.clear()

            ret = True

        self.thread_pool = None
        self.frame_index = 0
        return ret
import numpy as np
from rknnlite.api import RKNNLite
# from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

class RknnExecutor():
    def __init__(self, model_path:str, cores:tuple|int=0):
        self.model_path = model_path
        if isinstance(cores, int):
            self.core = cores
        else:
            self.core = cores[0]

        self.rknn_lite = None

    def init_rknn(self) -> RKNNLite:
        rknn_lite = RKNNLite()
        rknn_lite.load_rknn(self.model_path)

        if self.core == 0:
            mask = rknn_lite.NPU_CORE_0
        elif self.core == 1:
            mask = rknn_lite.NPU_CORE_1
        elif self.core == 2:
            mask = rknn_lite.NPU_CORE_2
        elif self.core == -1:
            mask = rknn_lite.NPU_CORE_ALL

        rknn_lite.init_runtime(core_mask=mask)
        self.rknn_lite = rknn_lite
    
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        """
        Inference with blocking.

        Args:
            input_data: a list of ndarrays
            input_format: 'nhwc' or 'nchw'
                - input_tensor should [n, h, w, c] or [n, c, h, w].
                - recommended [n, h, w, c].

        Returns:
            a list of ndarrays
        """

        if self.rknn_lite is None:
            self.init_rknn()

        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(tensor, (0, 2, 3, 1)) for tensor in input_data] # NCHW -> NHWC

        return self.rknn_lite.inference(input_data)

    def release(self):
        if self.rknn_lite is not None:
            self.rknn_lite.release()
            self.rknn_lite = None

            print("RKNN Executer released")


class RknnThreadPool():
    def __init__(self, model_path:str, cores:tuple[int]=(0, 1)):
        self.model_path = model_path
        self.cores = cores

        self.thread_num = len(self.cores)
        self.thread_pool = None

        # self.queue = deque()
        self.queue_list:list[Future] = []
        
        self.frame_index = 0

    def init_rknn_lite_threadpool(self) -> tuple[ThreadPoolExecutor, list[RKNNLite]]:
        rknn_list = []

        for core in self.cores:
            rknn_lite = RKNNLite()
            rknn_lite.load_rknn(self.model_path)

            if core == 0:
                mask = rknn_lite.NPU_CORE_0
            elif core == 1:
                mask = rknn_lite.NPU_CORE_1
            elif core == 2:
                mask = rknn_lite.NPU_CORE_2
            elif core == -1:
                mask = rknn_lite.NPU_CORE_ALL

            rknn_lite.init_runtime(core_mask=mask)
            rknn_list.append(rknn_lite)

        thread_pool = ThreadPoolExecutor(max_workers=self.thread_num, thread_name_prefix='rknn_thread_pool')

        return thread_pool, rknn_list

    @staticmethod
    def rknn_inference(rknn_lite:RKNNLite, input_list:list[np.ndarray]) -> list[np.ndarray]|None:
        outputs:list = rknn_lite.inference(input_list)
        return outputs

    def queue_put(self, frame, allow_drop:bool=True) -> None:
        if allow_drop and len(self.queue_list) >= self.thread_num:
            if not self.queue_list[0].done():
                return None
        
        excutor_index = self.frame_index
        
        # self.queue.append(self.thread_pool.submit(self.rknn_inference, self.rknn_list[excutor_num], frame))
        self.queue_list.append(self.thread_pool.submit(self.rknn_inference, self.rknn_list[excutor_index], frame))

        if self.frame_index >= self.thread_num -1 :
            self.frame_index = 0
        else:
            self.frame_index += 1

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> None:
        """
        Inference without blocking.
        
        Args:
            input_data: a list of ndarrays
            input_format: 'nhwc' or 'nchw'
                - input_tensor should [n, h, w, c] or [n, c, h, w].
                - recommended [n, h, w, c].
        """
        
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC
    
        if self.thread_pool is None:
            self.thread_pool, self.rknn_list = self.init_rknn_lite_threadpool()
            
            for i, core in enumerate(self.cores):
                self.queue_put(input_data, allow_drop=False)
                print(f'RKNN core: {core}, thread: {i}')

            if self.thread_num == 1:
                self.queue_put(input_data, allow_drop=False)

        else:
            self.queue_put(input_data)

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        if not self.queue_list:
            return None
        
        # future:Future = self.queue.popleft()
        future:Future = self.queue_list.pop(0)

        if block is False and future.done() is False:
            # self.queue.appendleft(future)
            self.queue_list.insert(0, future)
            return None

        result_list:list = future.result()

        return result_list

    def release(self) -> bool:
        ret = False

        if self.thread_pool is not None:
            for future in self.queue_list:
                future.cancel()
            self.queue_list.clear()

            self.thread_pool.shutdown(wait=True, cancel_futures=True)

            for i, rknn_lite in enumerate(self.rknn_list):
                rknn_lite.release()
                print(f'rknnlite: {i} released')
            self.rknn_list.clear()

            ret = True

        self.thread_pool = None
        self.frame_index = 0
        return ret

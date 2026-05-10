import os
import atexit
from pathlib import Path
import threading

import numpy as np
from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig



class QnnExecutor():
    def __init__(self, model_path:str):
        self.model_path = model_path

        self.qnn_context = None

    def init_qnn(self) -> QNNContext:
        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.BASIC)
        
        model_file = Path(self.model_path)
        model_name = model_file.stem
        self.qnn_context = QNNContext(model_name=model_name, model_path=self.model_path)

        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        if self.qnn_context is None:
            self.init_qnn()

        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC

        output = self.qnn_context.Inference(input_data)

        return output

    def release(self):
        if self.qnn_context is not None:
            #PerfProfile.RelPerfProfileGlobal()

            del self.qnn_context
            self.qnn_context = None
            
            print("QNN Executer released")



from concurrent.futures import ThreadPoolExecutor, Future

class QnnThreadPool():
    def __init__(self, model_path:str, cores:tuple[int]=(0, 1)):
        self.model_path = model_path
        self.cores = cores

        self.thread_num = len(self.cores)
        self.thread_pool = None

        self.queue_list:list[Future] = []
        
        self.frame_index = 0

    def init_qnn(self) -> list[QNNContext]:
        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)

        model_file = Path(self.model_path)
        model_name = model_file.stem

        qnn_list = []

        for core in self.cores:
            qnn_context = QNNContext(model_name=f'{model_name}_{core}', model_path=self.model_path)
            qnn_list.append(qnn_context)

        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        return qnn_list

    @staticmethod
    def thread_set_affinity() -> None:
        thread_id = threading.get_native_id()
        os.sched_setaffinity(thread_id, [4, 5, 6])
        print(f'Thread ID: {thread_id}')

    @staticmethod
    def qnn_inference(qnn_context:QNNContext, input_list:list[np.ndarray]) -> list[np.ndarray]|None:
        outputs:list = qnn_context.Inference(input_list)
        return outputs

    def queue_put(self, frame) -> None:
        excutor_index = self.frame_index
        self.queue_list.append(self.thread_pool.submit(self.qnn_inference, self.qnn_list[excutor_index], frame))

        if self.frame_index >= self.thread_num -1 :
            self.frame_index = 0
        else:
            self.frame_index += 1

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc', block_all_gets:bool=False) -> None:
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC

        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(self.thread_num, thread_name_prefix='qnn_thread_pool', initializer=self.thread_set_affinity)
            self.qnn_list = self.init_qnn()
            
            if block_all_gets is False:
                for i, core in enumerate(self.cores):
                    self.queue_put(input_data)
                    print(f'QNN core: {core}, thread: {i}')
            else:
                self.queue_put(input_data)
        else:
            self.queue_put(input_data)

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        future:Future = self.queue_list.pop(0)

        if block is False and future.done() is False:
            self.queue_list.insert(0, future)
            return None

        result_list:list = future.result()

        return result_list

    def release(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True, cancel_futures=True)

            for i in range(len(self.queue_list)):
                future = self.queue_list.pop(0)
                future.cancel()

            for i in range(len(self.qnn_list)):
                qnn_context = self.qnn_list.pop(0)
                del qnn_context
                print(f'qnn_context: {i} released')

            # PerfProfile.RelPerfProfileGlobal()

        self.thread_pool = None




from multiprocessing import Process, Pipe
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.connection import Connection

class ProcessQnnExecutor:
    def __init__(self, child_conn:Connection, model_path:str, in_shm_name:str, input_args_list:list):
        self.child_conn = child_conn
        self.model_path = model_path
        self.in_shm_name = in_shm_name
        self.input_args_list = input_args_list
        self.output_args_list:list|None = None

        self.pid = os.getpid()
        thread_id = threading.get_native_id()
        cpu_count = os.cpu_count()
        os.sched_setaffinity(thread_id, tuple(range(cpu_count // 2, cpu_count+1)))

        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)

        model_name =  os.path.splitext(os.path.basename(model_path))[0]
        self.qnn_context = QNNContext(model_name=f'{model_name}_{self.pid}', model_path=model_path, is_async=True)
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

        print(f"Process {self.pid}: Initialized QNNContext...")

        self.in_shm:SharedMemory = SharedMemory(name=in_shm_name) # 连接到共享内存
        self.out_shm:SharedMemory|None = None

        self.run()

    def run(self):
        while True:
            conn_get = self.child_conn.recv()

            if conn_get[0] is None:
                pass
            else:
                if conn_get[0] is True:
                    out_shm_name, self.output_args_list = conn_get[1], conn_get[2]
                    self.out_shm = SharedMemory(name=out_shm_name)
                else:
                    self.release()
                    break

            input_data = []
            for (input_shape, input_dtype, offset_range) in self.input_args_list:
                input_tensor = np.ndarray(input_shape, input_dtype, buffer=self.in_shm.buf[offset_range[0]:offset_range[1]]) # 创建 NumPy 数组视图
                input_data.append(input_tensor)
            
            output_list:list[np.ndarray] = self.qnn_context.Inference(input_data) # 执行推理
            
            if self.out_shm is not None:
                for i, (output_shape, output_dtype, offset_range) in enumerate(self.output_args_list):
                    shm_array = np.ndarray(output_shape, output_dtype, buffer=self.out_shm.buf[offset_range[0]:offset_range[1]]) # 创建 NumPy 数组视图
                    np.copyto(shm_array, output_list[i], casting='unsafe')
                output = (None,)

            else:
                output = (True, output_list)

            self.child_conn.send(output)

    def release(self):
        PerfProfile.RelPerfProfileGlobal()
        del self.qnn_context

        if self.in_shm is not None:
            self.in_shm.close()

        if self.out_shm is not None:
            self.out_shm.close()

        print(f'qnn_context process pid:{self.pid} released')
        exit(0)

class QnnProcessPool():
    def __init__(self, model_path:str, cores:tuple[int] = (0, 1)):
        self.model_path = model_path
        self.cores = tuple(cores)
        self.process_num = len(self.cores)

        self.process_list:list[Process]|None = None
        self.parent_conn_list:list[Connection]|None = None
        self.child_conn_list:list[Connection]|None= None

        self.shared_input_memory:SharedMemory|None = None
        self.shared_output_memory:SharedMemory|None = None

        self.inited_process_num = 0
        self.frame_index = 0

    @staticmethod
    def unlink_shm_at_exit(shm_name:str):
        try:
            shm = SharedMemory(name=shm_name)
            shm.unlink() # 取消链接，释放资源
            print(f"Shared memory block '{shm.name}' unlinked at exit")
        except FileNotFoundError:
            print(f"Shared memory block '{shm_name}' not found at exit")
            pass

    def init_shared_memory(self, array_list:list[np.ndarray]) -> tuple[SharedMemory, list[tuple[tuple[int,], np.dtype, tuple[int, int]]]]:
        shm_args_list:list[tuple[tuple[int,], np.dtype, tuple[int, int]]] = []
        total_byte_size = 0

        for array in array_list:
            offset = total_byte_size
            total_byte_size += array.nbytes

            offset_range = (offset, total_byte_size)
            shm_args_list.append((array.shape, array.dtype, offset_range))

        shared_memory = SharedMemory(create=True, size=total_byte_size) # 创建一个新的共享内存块
        atexit.register(self.unlink_shm_at_exit, shared_memory.name)

        print(f'created shared memory block:{shared_memory.name}, size:{total_byte_size}')
        return shared_memory, shm_args_list

    def copy_to_or_from_shared_memory(self, shared_memory:SharedMemory, shm_args_list:list[tuple[tuple[int],]], array_list:list[np.ndarray]|None=None) -> None|list[np.ndarray]:
        if array_list is not None:
            result_list = None
            for i, (array_shape, array_dtype, offset_range) in enumerate(shm_args_list):
                shared_array = np.ndarray(array_shape, array_dtype, buffer=shared_memory.buf[offset_range[0]:offset_range[1]])
                np.copyto(shared_array, array_list[i], casting='unsafe')
        else:
            result_list = []
            for i, (array_shape, array_dtype, offset_range) in enumerate(shm_args_list):
                shared_array = np.ndarray(array_shape, array_dtype, buffer=shared_memory.buf[offset_range[0]:offset_range[1]])
                result_list.append(shared_array)

        return result_list

    def init_qnn_process(self) -> None:
        self.process_list = []
        self.parent_conn_list = []
        self.child_conn_list = []

        model_file = Path(self.model_path)
        model_name = model_file.stem

        for i in range(self.process_num):
            process_name = f"{model_name}_QNN_process_{i}"
            parent_conn, child_conn = Pipe(duplex=True)

            process_args = (child_conn, self.model_path, self.shared_input_memory.name, self.input_args_list)
            Process_qnn = Process(target=ProcessQnnExecutor, name=process_name, args=process_args)
            Process_qnn.start()

            self.process_list.append(Process_qnn)
            self.parent_conn_list.append(parent_conn)
            self.child_conn_list.append(child_conn)


    def queue_put(self) -> None:
        input_args = (None,)

        persent_excutor_index = self.frame_index

        if self.shared_output_memory is not None and self.inited_process_num < self.process_num:
            input_args = (True, self.shared_output_memory.name, self.output_args_list)
            self.inited_process_num += 1

        self.parent_conn_list[persent_excutor_index].send(input_args) # 提交任务

        if self.frame_index >= self.process_num -1 :
            self.frame_index = 0
        else:
            self.frame_index += 1

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> None:
        """
        将任务放入进程池。
        """
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)).copy(order='C') for input_tensor in input_data]


        if self.shared_input_memory is None:
            self.shared_input_memory, self.input_args_list = self.init_shared_memory(input_data)

        if self.shared_input_memory is not None:
            self.copy_to_or_from_shared_memory(self.shared_input_memory, self.input_args_list, input_data)


        if self.process_list is None:
            self.init_qnn_process()
            
            for i in range(self.process_num):
                self.queue_put()

            if self.process_num == 1:
                self.queue_put()

        else:
            self.queue_put()

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        """
        从队列中获取一个任务的结果。
        """
        if not self.process_list:
            return None


        if block is True:
            results = self.parent_conn_list[self.frame_index].recv() # 阻塞直到任务完成
        else:
            results = None
            if self.parent_conn_list[self.frame_index].poll(): 
                results = self.parent_conn_list[self.frame_index].recv()

        output_list = None
        if results is not None:
            if self.shared_output_memory is not None:
                output_list = self.copy_to_or_from_shared_memory(self.shared_output_memory, self.output_args_list, None)

            elif results[0] is True:
                output_list = results[1]
                self.shared_output_memory, self.output_args_list = self.init_shared_memory(output_list)
                    

        return output_list


    def release(self) -> None:
        """
        释放所有资源，包括进程池和共享内存。
        """

        if self.process_list is not None:
            for i in range(self.process_num):
                self.parent_conn_list[i].send((False,))

        for shm in [self.shared_input_memory, self.shared_output_memory]:
            if shm is not None:
                shm.unlink()
                print(f"Shared memory block '{shm.name}' unlinked")

        if self.process_list is not None:
            for i in range(self.process_num):
                Process_qnn = self.process_list.pop(0)
                Process_qnn.join(timeout=0.1)
                Process_qnn.kill()
            

        if self.child_conn_list is not None:
            for i in range(self.process_num):
                child_conn = self.child_conn_list.pop(0)
                child_conn.close()
            
        if self.parent_conn_list is not None:
            for i in range(self.process_num):
                parent_conn = self.parent_conn_list.pop(0)
                parent_conn.close()
            
        self.shared_input_memory = None
        self.shared_output_memory = None
        self.process_list = None
        self.parent_conn_list = None
        self.child_conn_list = None

        self.frame_index = 0
        self.inited_process_num = 0    


import os
import re
import gc
import time
import atexit
import threading
from pathlib import Path
import numpy as np

from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig



def sanitize_name(name:str, replace_chars:str=r'()[]{}-\/:*?"<>|,') -> str:
    """将指定字符替换为 '_',并将连续下划线合并为一个"""
    trans_table = str.maketrans(replace_chars, '_' * len(replace_chars))
    name = name.translate(trans_table)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


class QnnExecutor():
    inited = False

    @classmethod
    def warm_npu(cls, model_path):
        if not cls.inited:
            QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
            tmp_model_name = str(time.monotonic_ns())
            qnn_context = QNNContext(model_name=tmp_model_name, model_path=model_path, is_async=True)
            del qnn_context
            gc.collect()
            cls.inited = True

    def __init__(self, model_path:str):
        self.model_path = model_path
        self.model_name = sanitize_name(Path(self.model_path).stem)
        self.model_name += f"_{int(time.monotonic_ns())}"
        #self.warm_npu(self.model_path)

        self.qnn_context = None

    def init_qnn(self):
        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
        
        self.qnn_context = QNNContext(model_name=self.model_name, model_path=self.model_path, is_async=True)
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        #SetQNNPerfProfileGlobal()

        print(f"QNNContext: {self.model_name} Initialized")
    
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        if self.qnn_context is None:
            self.init_qnn()

        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC

        try:
            output = self.qnn_context.Inference(input_data)
        except Exception as e:
            print(f"QNN Inference error: {e}")
            return None

        return output

    def release(self) -> bool:
        ret = False
        if self.qnn_context is not None:
            PerfProfile.RelPerfProfileGlobal()
            del self.qnn_context
            
            print(f'QNN Executer {self.model_name} released')
            ret = True

        self.qnn_context = None
        gc.collect()

        return ret



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
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

        model_name = sanitize_name(Path(self.model_path).stem)

        qnn_list = []

        for core in self.cores:
            qnn_context = QNNContext(model_name=f'{model_name}_{core}', model_path=self.model_path, is_async=True)
            qnn_list.append(qnn_context)

        return qnn_list

    @staticmethod
    def thread_set_affinity() -> None:
        if hasattr(os, 'sched_setaffinity'):
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


from multiprocessing import Process, Pipe, Queue, Manager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.connection import Connection
from multiprocessing.managers import SyncManager, DictProxy

from queue import Empty


def unlink_shm_at_exit(shm_name:str):
    try:
        shm = SharedMemory(name=shm_name)
        shm.unlink() # 取消链接,释放资源
        print(f"Shared memory block '{shm.name}' unlinked at process exit")
    except FileNotFoundError:
        # print(f"Shared memory block '{shm_name}' not found at exit")
        pass

def unlink_shm(shm_or_name:SharedMemory|str):
    shm_name = ""
    try:
        if isinstance(shm_or_name, SharedMemory):
            shm = shm_or_name
            shm_name = shm.name
            shm.close()
            shm.unlink()
        else:
            shm_name = shm_or_name
            shm = SharedMemory(name=shm_or_name)
            shm.close()
            shm.unlink()
            
        if shm_name:
            print(f"Shared memory block '{shm_name}' unlinked")

    except FileNotFoundError:
        # print(f"Shared memory block '{shm_or_name}' not found for unlinking")
        pass



def copy_listarray(src_list:list[np.ndarray], dst_list:list[np.ndarray]):
    """将 src_list 中的每个 NumPy 数组元素逐一复制到 dst_list 中对应位置的数组
    - src_list 和 dst_list 必须长度相同,且对应位置的数组形状和 dtype 兼容
    - 该函数执行元素级别的复制(np.copyto),而非简单的引用赋值
    """
    for src_array, dst_array in zip(src_list, dst_list):
        np.copyto(dst_array, src_array, casting='unsafe')

def get_shared_memory_view(shm:SharedMemory, shm_args_list:list[tuple[tuple[int],]]) -> list[np.ndarray]:
    shared_memory_views = []
    for (array_shape, array_dtype, offset_range) in shm_args_list:
        if isinstance(array_dtype, str):
            array_dtype = np.dtype(array_dtype)

        shared_array = np.ndarray(array_shape, array_dtype, buffer=shm.buf[offset_range[0]:offset_range[1]])
        shared_memory_views.append(shared_array)

    return shared_memory_views

def create_shared_memory(array_list:list[np.ndarray]) -> tuple[SharedMemory, list[tuple[tuple[int,], np.dtype, tuple[int, int]]]]:
    """仅创建共享内存块并返回 (shm, args_list),不写入数据调用方需自行 copy_listarray"""
    shared_memory:SharedMemory|None = None
    shm_args_list:list[tuple[tuple[int,], np.dtype, tuple[int, int]]] = []
    total_byte_size:int = 0

    for array in array_list:
        offset = total_byte_size
        total_byte_size += array.nbytes

        offset_range = (offset, total_byte_size)
        shm_args_list.append((array.shape, array.dtype, offset_range))

    if total_byte_size != 0:
        shared_memory = SharedMemory(create=True, size=total_byte_size) # 创建一个新的共享内存块
        atexit.register(unlink_shm_at_exit, shared_memory.name)

        all_mem_ranges = [[shape, offset_range] for shape, dtype, offset_range in shm_args_list]
        print(f'created shared memory block:{shared_memory.name}, size:{total_byte_size}, range:{all_mem_ranges}')

    else:
        shm_args_list = None

    return shared_memory, shm_args_list



class ProcessQnnExecutor:
    def __init__(self, child_conn:Connection, model_path:str, in_shm_name:str, input_args_list:list):
        self.child_conn = child_conn
        self.model_path = model_path
        self.in_shm_name = in_shm_name
        self.input_args_list = input_args_list
        self.output_args_list:list|None = None

        self.pid = os.getpid()
        if hasattr(os, 'sched_setaffinity'):
            try:
                thread_id = threading.get_native_id()
                cpu_count = os.cpu_count()
                os.sched_setaffinity(thread_id, tuple(range(cpu_count // 2, cpu_count+1)))
            except Exception as e:
                pass


        self.model_name = sanitize_name(Path(self.model_path).stem)
        self.model_name = f"{self.model_name}_{self.pid}"

        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
        self.qnn_context = QNNContext(model_name=self.model_name, model_path=model_path, is_async=True)
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)


        self.in_shm:SharedMemory|None = None
        self.out_shm:SharedMemory|None = None
        try:
            self.in_shm:SharedMemory = SharedMemory(name=in_shm_name) # 连接到共享内存
        except Exception as e:
            self.in_shm = None

        if self.in_shm is not None:
            print(f"QNNContext Process: {self.model_name} Initialized")
            self.run()

        self.release()

    def run(self):
        while True:
            conn_get = self.child_conn.recv()

            if conn_get[0] is None:
                pass
            else:
                if conn_get[0] is True:
                    out_shm_name, self.output_args_list = conn_get[1], conn_get[2]
                    if self.out_shm is not None and self.out_shm.name != out_shm_name:
                        unlink_shm(self.out_shm.name)

                    try:
                        self.out_shm = SharedMemory(name=out_shm_name)
                    except Exception as e:
                        return None
                else:
                    return None


            input_data:list[np.ndarray] = []
            try:
                for (input_shape, input_dtype, offset_range) in self.input_args_list:
                    input_tensor = np.ndarray(input_shape, input_dtype, buffer=self.in_shm.buf[offset_range[0]:offset_range[1]]) # 创建 NumPy 数组视图
                    input_data.append(input_tensor)
            except Exception as e:
                return None
            

            try:
                output_list:list[np.ndarray] = self.qnn_context.Inference(input_data) # 执行推理
            except Exception as e:
                print(f"QNN Inference error in process {self.pid}: {e}")
                output_list = []
                continue

            
            if output_list:
                if self.out_shm is not None:
                    for i, (output_shape, output_dtype, offset_range) in enumerate(self.output_args_list):
                        try:
                            shm_array = np.ndarray(output_shape, output_dtype, buffer=self.out_shm.buf[offset_range[0]:offset_range[1]]) # 创建 NumPy 数组视图
                        except Exception as e:
                            return None
                        
                        np.copyto(shm_array, output_list[i], casting='unsafe')

                    output = (None,)
                else:
                    self.out_shm, self.output_args_list = create_shared_memory(output_list)
                    out_view = get_shared_memory_view(self.out_shm, self.output_args_list)
                    copy_listarray(output_list, out_view)
                    output = (True, self.out_shm.name, self.output_args_list)
            else:
                output = None

            self.child_conn.send(output)

    def release(self):
        del self.qnn_context
        self.qnn_context = None
        gc.collect()

        if self.in_shm is not None:
            self.in_shm.close()

        if self.out_shm is not None:
            self.out_shm.close()

        print(f'qnn_context process {self.model_name} released')
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

    def init_qnn_process(self) -> None:
        self.process_list = []
        self.parent_conn_list = []
        self.child_conn_list = []

        model_name = Path(self.model_path).stem

        for i in range(self.process_num):
            process_name = f"{model_name}_QNN_process_{i}"
            parent_conn, child_conn = Pipe(duplex=True)

            process_args = (child_conn, self.model_path, self.shared_input_memory.name, self.input_args_list)
            Process_qnn = Process(target=ProcessQnnExecutor, name=process_name, args=process_args)
            Process_qnn.start()

            atexit.register(Process_qnn.kill)

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

        self.frame_index = (self.frame_index + 1) % self.process_num

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> None:
        """
        将任务放入进程池
        """
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)).copy(order='C') for input_tensor in input_data]


        if self.shared_input_memory is not None:
            input_view = get_shared_memory_view(self.shared_input_memory, self.input_args_list)
            copy_listarray(input_data, input_view)
        else:
            self.shared_input_memory, self.input_args_list = create_shared_memory(input_data)
            input_view = get_shared_memory_view(self.shared_input_memory, self.input_args_list)
            copy_listarray(input_data, input_view)

        if self.process_list is None:
            self.init_qnn_process()
            
            for i in range(self.process_num):
                self.queue_put()

            # if self.process_num == 1:
            #     self.queue_put()

        else:
            self.queue_put()

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        """
        从队列中获取一个任务的结果
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
                output_list = get_shared_memory_view(self.shared_output_memory, self.output_args_list)

            elif results[0] is True:
                shared_output_memory_name, output_args_list = results[1], results[2]
                shared_output_memory = SharedMemory(name=shared_output_memory_name)

                output_list = get_shared_memory_view(shared_output_memory, output_args_list)
                output_list = [array.copy() for array in output_list] # 从共享内存复制到独立内存,避免后续被覆盖

                self.shared_output_memory, self.output_args_list = create_shared_memory(output_list)
                out_view = get_shared_memory_view(self.shared_output_memory, self.output_args_list)
                copy_listarray(output_list, out_view)
                shared_output_memory.close()
                    

        return output_list


    def release(self) -> bool:
        """
        释放所有资源,包括进程池和共享内存
        """

        if self.child_conn_list is not None:
            for child_conn in self.child_conn_list:
                if child_conn.poll():
                    child_conn.recv()

        if self.process_list is not None:
            for parent_conn in self.parent_conn_list:
                parent_conn.send((False,))
                if parent_conn.poll():
                    parent_conn.recv()

        if self.process_list is not None:
            for i in range(self.process_num):
                Process_qnn = self.process_list.pop(0)
                Process_qnn.join(timeout=0.1)
            
        if self.child_conn_list is not None:
            for i in range(self.process_num):
                child_conn = self.child_conn_list.pop(0)
                child_conn.close()
            
        if self.parent_conn_list is not None:
            for i in range(self.process_num):
                parent_conn = self.parent_conn_list.pop(0)
                parent_conn.close()
            
        for shm in [self.shared_input_memory, self.shared_output_memory]:
            if shm is not None:
                unlink_shm(shm)

        self.shared_input_memory = None
        self.shared_output_memory = None
        self.process_list = None
        self.parent_conn_list = None
        self.child_conn_list = None

        self.frame_index = 0
        self.inited_process_num = 0 

        return True 


class ProcessQnnExecutor2:
    """Queue 版子进程执行器：用 parent_queue 收任务,child_queue 返结果

    manager_dict: multiprocessing.Manager().dict() 代理 —— 子进程可读写
    - 读取：查找已注册的 out_shm 名称及 output_args_list
    - 写入：首次创建 out_shm 时直接注册到 manager_dict(无需经主进程中转)
    """
    def __init__(self, parent_queue:Queue, child_queue:Queue, model_path:str, in_shm_name:str, input_args_list:list, manager_dict):
        self.parent_queue = parent_queue
        self.child_queue = child_queue
        self.model_path = model_path

        self.manager_dict:DictProxy = manager_dict
        self.output_args_list:list|None = None

        self.pid = os.getpid()
        try:
            thread_id = threading.get_native_id()
            cpu_count = os.cpu_count()
            os.sched_setaffinity(thread_id, tuple(range(cpu_count // 2, cpu_count+1)))
        except Exception as e:
            pass


        self.model_name = sanitize_name(Path(self.model_path).stem)
        self.model_name = f"{self.model_name}_{self.pid}"

        QNNConfig.Config('None', Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
        self.qnn_context = QNNContext(model_name=self.model_name, model_path=model_path, is_async=True)
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)


        self.in_shm:SharedMemory|None = None
        self.out_shm:SharedMemory|None = None
        self.input_array_list:list[np.ndarray]|None = None
        self.output_array_list:list[np.ndarray]|None = None

        try:
            self.in_shm:SharedMemory = SharedMemory(name=in_shm_name) # 连接到共享内存
            self.input_array_list = get_shared_memory_view(self.in_shm, input_args_list)
        except Exception as e:
            self.in_shm = None

        if self.in_shm is not None:
            print(f"QNNContext Process: {self.model_name} Initialized")
            self.run()

        self.release()

    def lookup_out_shm(self) -> bool:
        """从 manager_dict 查找已注册的 out_shm
        找到则打开(或复用)并恢复 output_args_list,返回 True；未找到返回 False
        """
        try:
            out_shm_name = self.manager_dict.get('out_shm_name')
            stored_args = self.manager_dict.get('output_args_list')
        except Exception:
            return False

        if not out_shm_name or not stored_args:
            return False

        # 已是同一个 shm → 无需重复操作
        if self.out_shm is not None:
            if self.out_shm.name == out_shm_name:
                return True

            else:
                # 关闭旧的 out_shm(名称不同时)
                unlink_shm(self.out_shm)
                self.out_shm = None

        try:
            self.out_shm = SharedMemory(name=out_shm_name)
        except Exception:
            self.out_shm = None
            return False

        # 恢复 output_args_list
        self.output_args_list = stored_args

        return True

    def register_out_shm_to_manager(self, out_shm:SharedMemory, output_args_list:list) -> None:
        """将当前 out_shm 及 output_args_list 注册到 manager_dict
        子进程在首次创建 out_shm 时调用,使主进程和其他子进程可发现
        dtype 序列化为字符串以确保跨进程兼容
        """
        serializable_args = [(shape, str(dtype), offset_range)
                             for shape, dtype, offset_range in output_args_list]
        
        self.manager_dict['out_shm_name'] = out_shm.name
        self.manager_dict['output_args_list'] = serializable_args

    def run(self):
        while True:
            conn_get = self.parent_queue.get()

            if conn_get is True:
                pass  # 正常推理信号
            else:
                return None # False → 关闭信号


            try:
                output_list:list[np.ndarray] = self.qnn_context.Inference(self.input_array_list) # 执行推理
            except Exception as e:
                print(f"QNN Inference error in process {self.pid}: {e}")
                output_list = []
                continue


            if output_list:
                if self.output_array_list is None:
                    # 尝试从 Manager 查找已有的 out_shm(其他子进程可能已创建)
                    if self.lookup_out_shm():
                        pass
                    else:
                        # 首个完成的子进程：创建 out_shm,注册到 manager
                        self.out_shm, self.output_args_list = create_shared_memory(output_list)
                        self.register_out_shm_to_manager(self.out_shm, self.output_args_list)

                    self.output_array_list = get_shared_memory_view(self.out_shm, self.output_args_list)


                # 复用已有的共享内存视图,直接复制数据
                copy_listarray(output_list, self.output_array_list)
                output = True

            else:
                output = None

            self.child_queue.put(output)

    def release(self):
        del self.qnn_context
        self.qnn_context = None
        gc.collect()

        self.child_queue.close()
        self.child_queue.cancel_join_thread()
            
        self.parent_queue.close()
        self.parent_queue.cancel_join_thread()

        if self.in_shm is not None:
            self.input_array_list.clear()
            self.in_shm.close()

        if self.out_shm is not None:
            self.output_array_list.clear()
            unlink_shm(self.out_shm)

        print(f'qnn_context process {self.model_name} released')
        exit(0)

class QnnProcessPool2():
    """Queue 版进程池：用 parent_queue 发任务,child_queue 收结果

    使用 multiprocessing.Manager().dict() 管理输出共享内存：
    - 子进程首次创建 out_shm 时直接写入 manager_dict
    - 主进程在 get() 中按需(output_array_list 为空时)从 manager_dict 查找并创建视图
    - 后续帧零拷贝返回视图
    """
    def __init__(self, model_path:str, cores:tuple[int] = (0, 1)):
        self.model_path = model_path
        self.cores = tuple(cores)
        self.process_num = len(self.cores)

        self.process_list:list[Process]|None = None
        self.parent_queue_list:list[Queue]|None = None
        self.child_queue_list:list[Queue]|None = None

        self.shared_input_memory:SharedMemory|None = None
        self.shared_output_memory:SharedMemory|None = None
        self.input_array_list:list[np.ndarray]|None = None
        self.output_array_list:list[np.ndarray]|None = None

        self.frame_index = 0
        self.inited_process_num = 0

        self.manager = Manager()
        self.manager_dict = self.manager.dict()

    def init_qnn_process(self, input_array_list:list[np.ndarray]) -> None:
        self.process_list = []
        self.parent_queue_list = []
        self.child_queue_list = []

        # 创建 NumPy 视图列表,供子进程推理时使用
        self.shared_input_memory, input_args_list = create_shared_memory(input_array_list)
        self.input_array_list = get_shared_memory_view(self.shared_input_memory, input_args_list)
        copy_listarray(input_array_list, self.input_array_list) # 将输入数据复制到共享内存

        model_name = Path(self.model_path).stem
        in_shm_name = self.shared_input_memory.name

        for i in range(self.process_num):
            process_name = f"{model_name}_QNN_process_{i}"
            parent_queue = Queue()
            child_queue = Queue()

            process_args = (parent_queue, child_queue, self.model_path, in_shm_name, input_args_list, self.manager_dict)
            Process_qnn = Process(target=ProcessQnnExecutor2, name=process_name, args=process_args)
            Process_qnn.start()

            atexit.register(Process_qnn.kill)

            self.process_list.append(Process_qnn)
            self.parent_queue_list.append(parent_queue)
            self.child_queue_list.append(child_queue)


    def queue_put(self) -> None:
        """向子进程发送推理信号,只发送 True
        子进程自行从 manager_dict 查找 out_shm
        """
        idx = self.frame_index
        self.parent_queue_list[idx].put(True)
        self.frame_index = (self.frame_index + 1) % self.process_num

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> None:
        """
        将任务放入进程池
        """
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)).copy(order='C') for input_tensor in input_data]


        if self.input_array_list is not None:
            copy_listarray(input_data, self.input_array_list)


        if self.process_list is None:
            self.init_qnn_process(input_data)
            
            for i in range(self.process_num):
                self.queue_put()
                self.inited_process_num += 1

        else:
            self.queue_put()

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        """从子进程获取推理结果

        子进程回复 True(有结果)或 None(推理失败)
        主进程按需从 manager_dict 懒加载 output_array_list:
        - 首次(output_array_list 为 None)→ 查 manager_dict,有则创建视图并返回独立副本
        - 后续 → 直接返回零拷贝视图(子进程已写入)
        - manager_dict 无 out_shm 信息 → 返回 None
        """
        if not self.process_list:
            return None

    
        if block is True:
            results = self.child_queue_list[self.frame_index].get() # 阻塞直到任务完成
        else:
            results = None
            try:
                results = self.child_queue_list[self.frame_index].get_nowait()
            except Empty:
                pass

        if not results:
            return None


        output_list = None
        if self.output_array_list is None:
            # 首次：从 manager_dict 查找并创建 output_array_list
            out_shm_name = self.manager_dict.get('out_shm_name')
            output_args_list = self.manager_dict.get('output_args_list')

            if out_shm_name is not None and output_args_list is not None:
                self.shared_output_memory = SharedMemory(name=out_shm_name)
                self.output_array_list = get_shared_memory_view(self.shared_output_memory, output_args_list)

                output_list = self.output_array_list

        else:
            # 后续帧：零拷贝返回视图
            output_list = self.output_array_list

        if self.inited_process_num <= self.process_num and output_list:
            # 首次返回独立副本,避免调用方意外修改共享内存
            output_list = [array.copy() for array in output_list]

        return output_list


    def release(self) -> bool:
        """
        释放所有资源,包括进程池、共享内存和 Manager
        """

        if self.parent_queue_list is not None:
            for parent_queue in self.parent_queue_list:
                parent_queue.put(False)

        if self.process_list is not None:
            for i in range(len(self.process_list)):
                Process_qnn = self.process_list.pop(0)
                Process_qnn.join(timeout=0.1)

            self.process_list = None
            

        if self.child_queue_list is not None:
            for i in range(len(self.child_queue_list)):
                child_queue = self.child_queue_list.pop(0)
                child_queue.close()
                child_queue.cancel_join_thread()
            self.child_queue_list = None

        if self.parent_queue_list is not None:
            for i in range(len(self.parent_queue_list)):
                parent_queue = self.parent_queue_list.pop(0)
                parent_queue.close()
                parent_queue.cancel_join_thread()
            self.parent_queue_list = None

        
        if self.input_array_list is not None:
            self.input_array_list.clear()
            self.input_array_list = None

        if self.output_array_list is not None:
            self.output_array_list.clear()
            self.output_array_list = None

        for shm in [self.shared_input_memory, self.shared_output_memory]:
            if shm is not None:
                unlink_shm(shm)

        self.shared_input_memory = None
        self.shared_output_memory = None

        self.manager.shutdown()

        self.frame_index = 0
        self.inited_process_num = 0

        return True 


class QnnExecutor2(QnnProcessPool):
    def __init__(self, model_path:str):
        super().__init__(model_path, cores=(0,))

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        super().put(input_data, input_format)
        output = self.get(block=True)
        return output

import os
import re
import time
import atexit
import platform
import threading
from pathlib import Path

from multiprocessing.connection import Connection
from multiprocessing.managers import SyncManager, DictProxy
from multiprocessing import Process, Pipe, Manager
from multiprocessing.shared_memory import SharedMemory



import numpy as np
from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig
from qai_appbuilder import QNNContextProc, QNNShareMemory



def check_arm_perf_cores() -> list[int]:
    """
    Check if the current system is Linux and ARM, 
    and return the core num of performance CPU cores.
    """
    
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "linux" or machine not in ("aarch64", "arm64", "armv7l", "armv6l"):
        return None

    # 2. 读取每个 CPU 核心的 capacity
    cpu_capacities: dict[int, int] = {}
    cpu_dir = Path("/sys/devices/system/cpu")

    for cpu_path in sorted(cpu_dir.glob("cpu[0-9]*")):
        try:
            core_id = int(cpu_path.name.removeprefix("cpu"))
            cap_file = cpu_path / "cpu_capacity"
            cpu_capacities[core_id] = int(cap_file.read_text().strip())
        except (ValueError, OSError):
            pass

    if not cpu_capacities:
        return None

    min_capacity = min(cpu_capacities.values())
    perf_cores_list = [k for k, v in cpu_capacities.items() if v > min_capacity]

    return perf_cores_list

def set_cpu_affinity(cores_list:list[int], pid:int|None=None):
    try:
        if pid is None:
            pid = threading.get_native_id()

        if cores_list:
            os.sched_setaffinity(pid, cores_list)
            print(f"Set CPU affinity to {cores_list} for PID {pid}")
    except Exception:
        pass

def sanitize_name(name:str, replace_chars:str=r'()[]{}-\/:*?"<>|,') -> str:
    """将指定字符替换为 '_',并将连续下划线合并为一个"""
    trans_table = str.maketrans(replace_chars, '_' * len(replace_chars))
    name = name.translate(trans_table)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def count_qnn_output_size(qnn_context:QNNContext|QNNContextProc) -> int:
    total_bytes = 0
    out_shape_list = qnn_context.getOutputShapes()
    out_dtype_list = qnn_context.getOutputDataType()

    for dtype_str, shape in zip(out_dtype_list, out_shape_list):
        # 从 dtype 字符串末尾提取位数, ufp8→8, fp16→16 → bytes = bits//8
        bits = int(''.join(c for c in dtype_str if c.isdigit()))
        elem_size = bits // 8
        num_elems = 1
        for dim in shape:
            num_elems *= dim
        tensor_bytes = num_elems * elem_size
        total_bytes += tensor_bytes

    return total_bytes


class QnnExecutor():
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.model_name = sanitize_name(Path(self.model_path).stem)
        self.model_name += f"_{int(time.monotonic_ns())}"

        self.qnn_context = None

    def init_qnn(self):
        QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
        
        self.qnn_context = QNNContext(self.model_name, self.model_path, is_async=True)
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

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
            #PerfProfile.RelPerfProfileGlobal()
            self.qnn_context.release()
            self.qnn_context = None
            
            print(f'QNN Executer {self.model_name} released')
            ret = True

        return ret


class QnnTaskExecutor():
    perf_cpu_cores = check_arm_perf_cores()

    def __init__(self, model_path:str):
        self.model_path = model_path

        self.model_name = sanitize_name(Path(self.model_path).stem)

        self.qnn_context_proc = None
        self.in_shm:QNNShareMemory|None = None

    def init_qnn(self, input_array_list:list[np.ndarray]):
        set_cpu_affinity(self.perf_cpu_cores)

        QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)

        time_stamp = int(time.monotonic())
        model_name = f"{self.model_name}_{time_stamp}"

        process_name = f"{model_name}_proc"
        self.qnn_context_proc = QNNContextProc(self.model_name, process_name, self.model_path)
        
        total_input_bytes = sum(arr.size * 4 for arr in input_array_list)
        total_output_bytes = count_qnn_output_size(self.qnn_context_proc)
        total_bytes = total_input_bytes + total_output_bytes

        self.in_shm = QNNShareMemory(f"{model_name}_in_shm", total_bytes)
        
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        print(f"QNNContextProc: {process_name} Initialized")
    
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        if self.qnn_context_proc is None:
            child_thread = threading.Thread(name="QNN_Task_Thread", target=self.init_qnn, args=(input_data,), daemon=True)
            child_thread.start()
            child_thread.join()

        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC

        try:
            output = self.qnn_context_proc.Inference(self.in_shm, input_data)
        except Exception as e:
            print(f"QNN Inference error: {e}")
            return None

        return output

    def release(self) -> bool:
        ret = False
        if self.qnn_context_proc is not None:
            #PerfProfile.RelPerfProfileGlobal()
            self.qnn_context_proc.release()
            self.in_shm.release()

            self.qnn_context_proc = None
            self.in_shm = None
            
            print(f'QNNContextProc: {self.model_name} released')
            ret = True

        return ret




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
    """Pipe 版子进程执行器：用 child_conn 收信号，child_conn 返结果

    manager_dict: multiprocessing.Manager().dict() 代理 —— 子进程可读写
    - 读取：查找已注册的 out_shm 名称及 output_args_list
    - 写入：首次创建 out_shm 时直接注册到 manager_dict（无需经主进程中转）
    """
    perf_cpu_cores = check_arm_perf_cores()

    def __init__(self, child_conn, model_path, in_shm_name:str, input_args_list:list, manager_dict, out_info_name):
        self.child_conn:Connection = child_conn
        self.model_path:str = model_path

        self.manager_dict:DictProxy = manager_dict
        self.out_info_name:str = out_info_name
        self.output_args_list:list|None = None

        self.pid = os.getpid()
        set_cpu_affinity(self.perf_cpu_cores, self.pid)

        self.model_name = sanitize_name(Path(self.model_path).stem)
        self.model_name_with_pid = f"{self.model_name}_{self.pid}"

        QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
        self.qnn_context = QNNContext(self.model_name_with_pid, model_path, is_async=True)
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
            print(f"QNNContext Process: {self.model_name_with_pid} Initialized")
            self.run()

        self.release()

    def lookup_out_shm(self) -> bool:
        """从 manager_dict 查找已注册的 out_shm
        找到则打开（或复用）并恢复 output_args_list，返回 True；未找到返回 False
        """
        try:
            out_shm_info = self.manager_dict.get(self.out_info_name)
        except Exception:
            return False

        if not out_shm_info:
            return False

        out_shm_name, stored_args = out_shm_info

        # 已是同一个 shm → 无需重复操作
        if self.out_shm is not None:
            if self.out_shm.name == out_shm_name:
                return True
            else:
                self.output_array_list = None
                unlink_shm(self.out_shm)
                self.out_shm = None

        try:
            self.out_shm = SharedMemory(name=out_shm_name)
        except Exception:
            self.out_shm = None
            return False

        self.output_args_list = stored_args
        return True

    def register_out_shm_to_manager(self, out_shm:SharedMemory, output_args_list:list) -> None:
        """将当前 out_shm 及 output_args_list 注册到 manager_dict
        子进程在首次创建 out_shm 时调用，使主进程和其他子进程可发现
        dtype 序列化为字符串以确保跨进程兼容
        使用单键元组写入，保证原子性
        """
        serializable_args = [(shape, str(dtype), offset_range)
                             for shape, dtype, offset_range in output_args_list]
        self.manager_dict[self.out_info_name] = (out_shm.name, serializable_args)

    def run(self):
        while True:
            conn_get = self.child_conn.recv()

            if conn_get is True:
                pass  # 正常推理信号
            else:
                return None  # False → 关闭信号


            try:
                output_list:list[np.ndarray] = self.qnn_context.Inference(self.input_array_list) # 执行推理
            except Exception as e:
                print(f"QNN Inference error in process {self.pid}: {e}")
                output_list = []
                continue


            if output_list:
                if self.output_array_list is None:
                    # 尝试从 Manager 查找已有的 out_shm（其他子进程可能已创建）
                    if self.lookup_out_shm():
                        pass
                    else:
                        # 首个完成的子进程：创建 out_shm，注册到 manager
                        self.out_shm, self.output_args_list = create_shared_memory(output_list)
                        self.register_out_shm_to_manager(self.out_shm, self.output_args_list)

                    self.output_array_list = get_shared_memory_view(self.out_shm, self.output_args_list)

                # 复用已有的共享内存视图，直接复制数据
                copy_listarray(output_list, self.output_array_list)
                output = True
            else:
                output = None

            self.child_conn.send(output)

    def release(self):
        self.qnn_context.release()
        self.qnn_context = None


        self.child_conn.close()

        if self.input_array_list is not None:
            self.input_array_list.clear()
            self.input_array_list = None

        if self.output_array_list is not None:
            self.output_array_list.clear()
            self.output_array_list = None

        if self.in_shm is not None:
            self.in_shm.close()

        if self.out_shm is not None:
            unlink_shm(self.out_shm)

        print(f'qnn_context process {self.model_name_with_pid} released')
        exit(0)

class QnnProcessPool():
    instance_num = 0
    manager:SyncManager|None = None
    manager_dict:DictProxy|None = None

    @classmethod
    def manage_dict_manager(cls, release:bool=False) -> DictProxy|None:
        if not release:
            if cls.manager is None:
                cls.manager = Manager()
                cls.manager_dict = cls.manager.dict()

            cls.instance_num += 1
            return cls.manager_dict
        
        else:
            cls.instance_num -= 1

            if cls.manager is not None and cls.instance_num <= 0:
                cls.manager.shutdown()
                cls.manager = None
                cls.manager_dict = None
                print("QnnProcessPool manager_dict released")

            return None


    def __init__(self, model_path:str, cores:tuple[int]=(0, 1)):
        self.model_path = model_path
        self.cores = tuple(cores)
        self.process_num = len(self.cores)

        self.model_name = str(Path(self.model_path).stem)
        self.out_info_name = f"{sanitize_name(self.model_name)}_out_shm_info"

        self.process_list:list[Process]|None = None
        self.parent_conn_list:list[Connection]|None = None
        self.child_conn_list:list[Connection]|None= None

        self.shared_input_memory:SharedMemory|None = None
        self.shared_output_memory:SharedMemory|None = None
        self.input_array_list:list[np.ndarray]|None = None
        self.output_array_list:list[np.ndarray]|None = None

        self.inited_process_num = 0
        self.frame_index = 0

    def init_qnn_process(self, input_array_list:list[np.ndarray]) -> tuple[list[Process], list[Connection], list[Connection]]:
        process_list:list[Process] = []
        parent_conn_list:list[Connection] = []
        child_conn_list:list[Connection] = []

        # 创建 NumPy 视图列表,供子进程推理时使用
        self.shared_input_memory, input_args_list = create_shared_memory(input_array_list)
        self.input_array_list = get_shared_memory_view(self.shared_input_memory, input_args_list)
        copy_listarray(input_array_list, self.input_array_list) # 将输入数据复制到共享内存

        manager_dict = self.manage_dict_manager()
        in_shm_name = self.shared_input_memory.name

        for i in range(self.process_num):
            process_name = f"{self.model_name}_QNN_process_{i}"
            parent_conn, child_conn = Pipe(duplex=True)

            process_args = (child_conn, self.model_path, in_shm_name, input_args_list, manager_dict, self.out_info_name)
            Process_qnn = Process(target=ProcessQnnExecutor, name=process_name, args=process_args, daemon=True)
            Process_qnn.start()

            atexit.register(Process_qnn.kill)

            process_list.append(Process_qnn)
            parent_conn_list.append(parent_conn)
            child_conn_list.append(child_conn)

        return process_list, parent_conn_list, child_conn_list

    def queue_put(self) -> None:
        """向子进程发送推理信号,只发送 True
        子进程自行从 manager_dict 查找 out_shm
        """
        idx = self.frame_index
        self.parent_conn_list[idx].send(True)
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
            self.process_list, self.parent_conn_list, self.child_conn_list = self.init_qnn_process(input_data)

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
            results = self.parent_conn_list[self.frame_index].recv() # 阻塞直到任务完成
        else:
            results = None
            if self.parent_conn_list[self.frame_index].poll(): 
                results = self.parent_conn_list[self.frame_index].recv()

        if not results:
            return None

        output_list = None
        if self.output_array_list is None:
            # 首次：从 manager_dict 查找并创建 output_array_list
            out_shm_info = self.manager_dict.get(self.out_info_name)

            if out_shm_info is not None:
                out_shm_name, output_args_list = out_shm_info
                self.shared_output_memory = SharedMemory(name=out_shm_name)
                self.output_array_list = get_shared_memory_view(self.shared_output_memory, output_args_list)

                output_list = self.output_array_list

        else:
            # 后续帧：零拷贝返回视图
            output_list = self.output_array_list

        if self.inited_process_num <= self.process_num and output_list:
            # 首轮返回独立副本，避免调用方意外修改共享内存
            output_list = [array.copy() for array in output_list]

        return output_list


    def release(self) -> bool:
        """
        释放所有资源,包括进程池、共享内存和 Manager
        """

        if self.parent_conn_list is not None:
            for parent_conn in self.parent_conn_list:
                parent_conn.send(False)

        if self.process_list is not None:
            for i in range(len(self.process_list)):
                Process_qnn = self.process_list.pop(0)
                Process_qnn.join()
            self.process_list = None


        if self.child_conn_list is not None:
            for child_conn in self.child_conn_list:
                child_conn.close()
            self.child_conn_list.clear()
            self.child_conn_list = None

        if self.parent_conn_list is not None:
            for parent_conn in self.parent_conn_list:
                parent_conn.close()
            self.parent_conn_list.clear()
            self.parent_conn_list = None


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

        self.manage_dict_manager(release=True)

        self.frame_index = 0
        self.inited_process_num = 0 

        return True 


class QnnExecutor3(QnnProcessPool):
    def __init__(self, model_path:str):
        super().__init__(model_path, cores=(0,))

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]|None:
        super().put(input_data, input_format)
        output = self.get(block=True)
        return output



class QnnTaskPool():
    perf_cpu_cores = check_arm_perf_cores()

    def __init__(self, model_path:str, cores:tuple[int]=(0, 1)):
        self.model_path = model_path
        self.cores = cores

        self.task_num = len(self.cores)
        self.frame_index = 0

        self.child_thread:threading.Thread|None = None
        self.qnn_proc_list:list[QNNContextProc]|None = None
        self.qnn_shm_list:list[QNNShareMemory]|None = None

        self.queue_list:list[tuple[int, str]] = []

    def init_qnn(self, input_array_list:list[np.ndarray]) -> tuple[list[QNNContextProc], list[QNNShareMemory]]:
        set_cpu_affinity(self.perf_cpu_cores)

        QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
    
        model_name = sanitize_name(Path(self.model_path).stem)
        qnn_process_list = []
        qnn_shm_list = []

        model_name = Path(self.model_path).stem
        total_input_bytes = sum(arr.size * 4 for arr in input_array_list)
        

        for i in range(self.task_num ):
            process_name = f"{model_name}_QNN_proc_{i}"
            qnn_process = QNNContextProc(model_name, process_name, self.model_path, is_async=True)

            total_output_bytes = count_qnn_output_size(qnn_process)
            total_bytes = total_input_bytes + total_output_bytes

            qnn_shm = QNNShareMemory(f"{process_name}_inshm", total_bytes)
            qnn_process_list.append(qnn_process)
            qnn_shm_list.append(qnn_shm)

        
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        self.qnn_proc_list = qnn_process_list
        self.qnn_shm_list = qnn_shm_list

        return qnn_process_list, qnn_shm_list

    def queue_put(self, frame) -> None:
        current_index = self.frame_index

        qnn_process = self.qnn_proc_list[current_index]
        qnn_shm = self.qnn_shm_list[current_index]
        qnn_task_id = qnn_process.InferenceAsync(qnn_shm, frame)

        self.queue_list.append((current_index, qnn_task_id))
        self.frame_index = (self.frame_index + 1) % self.task_num

    def put(self, input_data:list[np.ndarray], input_format:str='nhwc', block_all_gets:bool=False) -> None:
        if input_format == 'nhwc':
            pass
        elif input_format == 'nchw':
            input_data = [np.transpose(input_tensor, (0, 2, 3, 1)) for input_tensor in input_data] # NCHW -> NHWC

        if self.qnn_proc_list is None:
            self.child_thread = threading.Thread(name="QNN_Task_Thread", target=self.init_qnn, args=(input_data,), daemon=True)
            self.child_thread.start()
            self.child_thread.join()
            #self.qnn_proc_list, self.qnn_shm_list = self.init_qnn(input_data)
            
            if block_all_gets is False:
                for i, core in enumerate(self.cores):
                    self.queue_put(input_data)
                    print(f'QNN task: {core}, thread: {i}')
            else:
                self.queue_put(input_data)
        else:
            self.queue_put(input_data)

    def get(self, block:bool=True) -> list[np.ndarray]|None:
        # future:Future = self.queue_list.pop(0)

        # if block is False and future.done() is False:
        #     self.queue_list.insert(0, future)
        #     return None

        if len(self.queue_list) == 0:
            return None
        
        last_index, qnn_task_id = self.queue_list.pop(0)

        qnn_process = self.qnn_proc_list[last_index]
        qnn_shm = self.qnn_shm_list[last_index]
        result_list:list[np.ndarray] = qnn_process.InferenceWait(qnn_task_id, qnn_shm)

        return result_list

    def release(self) -> None:
        if self.qnn_proc_list is not None:
            for i in range(len(self.qnn_proc_list)):
                qnn_context_proc = self.qnn_proc_list.pop(0)
                qnn_context_proc.release()
  
                print(f'qnn_context task: {i} released')

        if self.qnn_shm_list is not None:
            for i in range((len(self.qnn_shm_list))):
                in_shm = self.qnn_shm_list.pop(0)
                in_shm.release()

        self.qnn_proc_list = None
        self.qnn_shm_list = None
        self.queue_list.clear()

        self.frame_index = 0
        self.child_thread = None

            # PerfProfile.RelPerfProfileGlobal()






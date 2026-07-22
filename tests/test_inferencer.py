"""Tests for Edge_Inferencer modules.

Tests cover code that can be exercised without actual hardware (RKNPU, QNN HTP).
Hardware-dependent modules (qnn_inferencer, rknn_inferencer) are conditionally
skipped when their SDK imports are unavailable.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Tests for ai_inferencer.py
# =============================================================================

class TestTemporarySysPath:
    """Test the temporary_sys_path context manager."""

    def test_adds_and_removes_path(self):
        from ai_inferencer import temporary_sys_path

        test_path = "/tmp/nonexistent_test_path_xyz"
        assert test_path not in sys.path

        with temporary_sys_path(test_path):
            assert test_path in sys.path

        assert test_path not in sys.path

    def test_does_not_remove_existing_path(self):
        from ai_inferencer import temporary_sys_path

        existing = sys.path[0]
        with temporary_sys_path(existing):
            assert existing in sys.path

        # Should still be there (wasn't removed)
        assert existing in sys.path

    def test_nested_contexts(self):
        from ai_inferencer import temporary_sys_path

        p1 = "/tmp/_test_path_1"
        p2 = "/tmp/_test_path_2"

        with temporary_sys_path(p1):
            assert p1 in sys.path
            with temporary_sys_path(p2):
                assert p1 in sys.path
                assert p2 in sys.path
            assert p1 in sys.path
            assert p2 not in sys.path

        assert p1 not in sys.path

    def test_exception_safety(self):
        from ai_inferencer import temporary_sys_path

        test_path = "/tmp/_test_exc_path"
        try:
            with temporary_sys_path(test_path):
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        assert test_path not in sys.path


class TestEmptyAIInferencer:
    """Test the EmptyAIInferencer null-object pattern."""

    def test_put_returns_none(self):
        from ai_inferencer import EmptyAIInferencer
        inf = EmptyAIInferencer()
        assert inf.put("anything") is None

    def test_get_returns_none(self):
        from ai_inferencer import EmptyAIInferencer
        inf = EmptyAIInferencer()
        assert inf.get("anything") is None

    def test_release_passes(self):
        from ai_inferencer import EmptyAIInferencer
        inf = EmptyAIInferencer()
        inf.release()  # Should not raise

    def test_init_passes(self):
        from ai_inferencer import EmptyAIInferencer
        inf = EmptyAIInferencer()
        assert isinstance(inf, EmptyAIInferencer)


class TestIdentifyModelType:
    """Test AIInferencer.identify_model_type static method."""

    def test_rknn_extension(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("model.rknn") == "rknn"

    def test_onnx_extension(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("model.onnx") == "onnx"

    def test_qnn_bin_extension(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("model.bin") == "qnn"

    def test_unknown_extension(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("model.tflite") == "Unknown"
        assert AIInferencer.identify_model_type("model.pt") == "Unknown"
        assert AIInferencer.identify_model_type("model") == "Unknown"

    def test_none_path(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type(None) == "Unknown"
        assert AIInferencer.identify_model_type("") == "Unknown"

    def test_case_insensitive(self):
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("model.RKNN") == "rknn"
        assert AIInferencer.identify_model_type("model.ONNX") == "onnx"

    def test_with_real_tempfile(self):
        from ai_inferencer import AIInferencer
        with tempfile.NamedTemporaryFile(suffix=".rknn", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            assert AIInferencer.identify_model_type(path) == "rknn"
        finally:
            os.unlink(path)

    def test_file_not_found_still_returns_ext(self):
        """Even when file doesn't exist, the extension is still detected."""
        from ai_inferencer import AIInferencer
        assert AIInferencer.identify_model_type("/nonexistent/model.rknn") == "rknn"


class TestAIInferencerInit:
    """Test AIInferencer base initialization (without hardware deps)."""

    def test_init_with_none_uses_empty_inferencer(self):
        """model_path=None should use EmptyAIInferencer."""
        from ai_inferencer import AIInferencer, EmptyAIInferencer
        inf = AIInferencer(None)
        assert isinstance(inf.inferfacer, EmptyAIInferencer)

    def test_init_with_unknown_uses_empty_inferencer(self):
        from ai_inferencer import AIInferencer, EmptyAIInferencer
        inf = AIInferencer("model.unknown")
        assert isinstance(inf.inferfacer, EmptyAIInferencer)

    def test_str_model_path(self):
        """model_path should be converted to string."""
        from ai_inferencer import AIInferencer
        inf = AIInferencer(None)
        assert isinstance(inf.model_path, str)


class TestTimeitDecorator:
    """Test the timeit decorator."""

    def test_basic_decorator(self):
        from ai_inferencer import timeit

        @timeit
        def simple_func():
            return 42

        assert simple_func() == 42
        assert simple_func.__name__ == "simple_func"

    def test_decorator_with_args(self):
        from ai_inferencer import timeit

        @timeit(measure_cycle_time=False)
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_decorator_cycle_time(self):
        from ai_inferencer import timeit

        @timeit(measure_cycle_time=True)
        def cycle_func():
            return "cycle"

        assert cycle_func() == "cycle"

    def test_preserves_signature(self):
        from ai_inferencer import timeit
        import inspect

        @timeit
        def func_with_docs(a, b=10):
            """My docstring."""
            return a + b

        sig = inspect.signature(func_with_docs)
        assert "a" in sig.parameters
        assert "b" in sig.parameters
        assert func_with_docs.__doc__ == "My docstring."
        assert func_with_docs(1, 2) == 3


# =============================================================================
# Tests for onnx_inferencer.py
# =============================================================================

class TestOnnxExecutor:
    """Test OnnxExecutor initialization and release."""

    def test_init_defaults(self):
        """OnnxExecutor should set defaults without loading a model."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor("dummy.onnx")
        assert exec.model_path == "dummy.onnx"
        assert exec.session is None
        assert exec.input_names == []
        assert exec.output_names == []
        assert exec.float_inputs is False
        assert exec.last_outputs is None

    def test_init_with_providers(self):
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor("dummy.onnx")
        assert "CPUExecutionProvider" in exec.providers

    def test_release_without_init(self):
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor("dummy.onnx")
        exec.release()
        assert exec.session is None

    def test_release_clears_names(self):
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor("dummy.onnx")
        exec.input_names = ["input1"]
        exec.output_names = ["output1"]
        exec.session = MagicMock()
        exec.release()
        assert exec.input_names == []
        assert exec.output_names == []


class TestOnnxExecutorWithRealModel:
    """Test OnnxExecutor with a minimal valid ONNX model."""

    @pytest.fixture(scope="class")
    def onnx_model_path(self):
        """Create a minimal valid ONNX Identity model for testing."""
        import onnx
        import io

        X = onnx.helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        Y = onnx.helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f)
            path = f.name

        yield path
        os.unlink(path)

    def test_init_onnx(self, onnx_model_path):
        """Should initialize session with a valid model file."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        exec.init_onnx()
        assert exec.session is not None
        assert len(exec.input_names) > 0
        assert len(exec.output_names) > 0
        exec.release()

    def test_put_nhwc_to_nchw(self, onnx_model_path):
        """NHWC input should be transposed to NCHW internally."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        input_data = [np.random.rand(1, 224, 224, 3).astype(np.float32)]
        outputs = exec.put(input_data, input_format="nhwc")
        assert outputs is not None
        assert outputs[0].shape == (1, 3, 224, 224)
        exec.release()

    def test_put_nchw_passthrough(self, onnx_model_path):
        """NCHW input should pass through without transpose."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        input_data = [np.random.rand(1, 3, 224, 224).astype(np.float32)]
        outputs = exec.put(input_data, input_format="nchw")
        assert outputs is not None
        assert outputs[0].shape == (1, 3, 224, 224)
        exec.release()

    def test_put_get_lifecycle(self, onnx_model_path):
        """put() then get() should return the same result."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        input_data = [np.random.rand(1, 224, 224, 3).astype(np.float32)]
        outputs = exec.put(input_data)
        retrieved = exec.get()
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved[0], outputs[0])
        # After get(), cache should be cleared
        assert exec.get() is None
        exec.release()

    def test_float_input_conversion(self, onnx_model_path):
        """Integer inputs should be auto-converted to float32."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        input_data = [np.ones((1, 224, 224, 3), dtype=np.int32)]
        outputs = exec.put(input_data, input_format="nhwc")
        assert outputs is not None
        exec.release()

    def test_multiple_inferences(self, onnx_model_path):
        """Running inference multiple times should work."""
        from onnx_inferencer import OnnxExecutor
        exec = OnnxExecutor(onnx_model_path)
        for _ in range(3):
            input_data = [np.random.rand(1, 224, 224, 3).astype(np.float32)]
            outputs = exec.put(input_data)
            assert outputs is not None
        exec.release()


# =============================================================================
# Tests for qnn_inferencer.py (hardware-independent functions only)
# =============================================================================

qnn_available = False
try:
    import qnn_inferencer  # noqa: F401
    qnn_available = True
except ModuleNotFoundError:
    qnn_available = False


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestSanitizeName:
    """Test sanitize_name helper function."""

    def test_replaces_special_chars(self):
        from qnn_inferencer import sanitize_name
        assert sanitize_name("test(name)") == "test_name"

    def test_replaces_brackets(self):
        from qnn_inferencer import sanitize_name
        result = sanitize_name("model[0]-v1.2")
        assert "_" in result

    def test_collapses_underscores(self):
        from qnn_inferencer import sanitize_name
        assert sanitize_name("a___b___c") == "a_b_c"

    def test_strips_trailing_underscores(self):
        from qnn_inferencer import sanitize_name
        result = sanitize_name("_test_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_clean_name_unchanged(self):
        from qnn_inferencer import sanitize_name
        assert sanitize_name("simple_name") == "simple_name"
        assert sanitize_name("abc123") == "abc123"

    def test_slash_and_colon(self):
        from qnn_inferencer import sanitize_name
        result = sanitize_name("path/to/model:v1")
        assert "_" in result
        assert "/" not in result
        assert ":" not in result


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestCopyListArray:
    """Test copy_listarray helper."""

    def test_copies_values(self):
        from qnn_inferencer import copy_listarray
        src = [np.array([1, 2, 3], dtype=np.float32)]
        dst = [np.array([0, 0, 0], dtype=np.float32)]
        copy_listarray(src, dst)
        np.testing.assert_array_equal(dst[0], [1, 2, 3])

    def test_multiple_arrays(self):
        from qnn_inferencer import copy_listarray
        src = [np.array([1.0]), np.array([2.0, 3.0])]
        dst = [np.array([0.0]), np.array([0.0, 0.0])]
        copy_listarray(src, dst)
        assert dst[0][0] == 1.0
        assert dst[1][0] == 2.0
        assert dst[1][1] == 3.0

    def test_different_shapes(self):
        from qnn_inferencer import copy_listarray
        src = [np.ones((3, 3), dtype=np.float64)]
        dst = [np.zeros((3, 3), dtype=np.float64)]
        copy_listarray(src, dst)
        np.testing.assert_array_equal(dst[0], np.ones((3, 3)))

    def test_empty_lists(self):
        from qnn_inferencer import copy_listarray
        copy_listarray([], [])  # Should not raise


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestCheckArmPerfCores:
    """Test check_arm_perf_cores."""

    def test_returns_none_on_non_arm(self):
        from qnn_inferencer import check_arm_perf_cores
        import platform
        machine = platform.machine().lower()
        if machine not in ("aarch64", "arm64", "armv7l", "armv6l"):
            assert check_arm_perf_cores() is None

    @patch("platform.machine", return_value="aarch64")
    @patch("platform.system", return_value="Linux")
    def test_returns_none_when_no_cpuinfo(self, mock_sys, mock_mach):
        from qnn_inferencer import check_arm_perf_cores
        assert check_arm_perf_cores() is None


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestSetCpuAffinity:
    """Test set_cpu_affinity."""

    def test_no_crash_no_cores(self):
        from qnn_inferencer import set_cpu_affinity
        set_cpu_affinity([])

    def test_no_crash_with_cores(self):
        from qnn_inferencer import set_cpu_affinity
        try:
            set_cpu_affinity([0, 1])
        except Exception:
            pass


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestUnlinkShm:
    """Test shared memory unlinking — handles missing gracefully."""

    def test_unlink_nonexistent(self):
        from qnn_inferencer import unlink_shm
        unlink_shm("_test_nonexistent_shm_xyz_")

    def test_unlink_at_exit_nonexistent(self):
        from qnn_inferencer import unlink_shm_at_exit
        unlink_shm_at_exit("_test_nonexistent_shm_xyz_")

    def test_unlink_with_non_string(self):
        from qnn_inferencer import unlink_shm
        unlink_shm(42)


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestCreateSharedMemory:
    """Test create_shared_memory."""

    def test_empty_list_returns_none(self):
        from qnn_inferencer import create_shared_memory
        shm, args_list = create_shared_memory([])
        assert shm is None
        assert args_list is None

    def test_creates_with_small_arrays(self):
        from qnn_inferencer import create_shared_memory
        arr = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        try:
            shm, args_list = create_shared_memory(arr)
            assert shm is not None
            assert args_list is not None
            assert len(args_list) == 1
        finally:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass


@pytest.mark.skipif(not qnn_available, reason="qai_appbuilder SDK not installed")
class TestGetSharedMemoryView:
    """Test get_shared_memory_view."""

    def test_with_created_shm(self):
        from qnn_inferencer import create_shared_memory, get_shared_memory_view
        arr = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
        try:
            shm, args_list = create_shared_memory(arr)
            assert shm is not None
            views = get_shared_memory_view(shm, args_list)
            assert len(views) == 1
            assert views[0].shape == (2, 2)
        finally:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass


# =============================================================================
# Tests for rknn_inferencer.py (basic init without hardware)
# =============================================================================

rknn_available = False
try:
    import rknn_inferencer  # noqa: F401
    rknn_available = True
except ModuleNotFoundError:
    rknn_available = False


@pytest.mark.skipif(not rknn_available, reason="rknnlite SDK not installed")
class TestRknnExecutor:
    """Test RknnExecutor init (no RKNPU hardware needed for init)."""

    def test_init_defaults(self):
        from rknn_inferencer import RknnExecutor
        exec = RknnExecutor("model.rknn", cores=(0,))
        assert exec.model_path == "model.rknn"
        assert exec.core == 0
        assert exec.rknn_lite is None

    def test_init_with_different_core(self):
        from rknn_inferencer import RknnExecutor
        exec = RknnExecutor("model.rknn", cores=(1,))
        assert exec.core == 1

    def test_release_without_init(self):
        from rknn_inferencer import RknnExecutor
        exec = RknnExecutor("model.rknn", cores=(0,))
        exec.release()


@pytest.mark.skipif(not rknn_available, reason="rknnlite SDK not installed")
class TestRknnThreadPool:
    """Test RknnThreadPool init (no RKNPU hardware needed)."""

    def test_init_defaults(self):
        from rknn_inferencer import RknnThreadPool
        pool = RknnThreadPool("model.rknn", cores=(0, 1))
        assert pool.thread_num == 2
        assert pool.frame_index == 0
        assert pool.thread_pool is None

    def test_init_single_core(self):
        from rknn_inferencer import RknnThreadPool
        pool = RknnThreadPool("model.rknn", cores=(0,))
        assert pool.thread_num == 1

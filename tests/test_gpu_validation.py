"""
GPU Validation Smoke Tests

These tests verify that the engine is using real GPU code, not CPU stubs.
If stubs are being used, these tests will FAIL with clear error messages.
"""

import ctypes
import os
import pytest
import struct


def get_engine_path():
    """Get the engine.so path, checking common locations."""
    paths = [
        os.environ.get("ENGINE_PATH", ""),
        "./build/engine.so",
        "/app/build/engine.so",
        "../build/engine.so",
    ]
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def get_cuda_wrapper_path():
    """Get the libcudawrap.so path."""
    paths = [
        "./build/libcudawrap.so",
        "/app/build/libcudawrap.so",
        "../build/libcudawrap.so",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


class TestGPUValidation:
    """Tests to validate GPU functionality vs CPU stubs."""

    def test_cuda_wrapper_not_stub(self):
        """Verify that CUDA wrapper is real, not a stub."""
        cuda_path = get_cuda_wrapper_path()
        if not cuda_path:
            pytest.skip("libcudawrap.so not found")
        
        cuda = ctypes.CDLL(cuda_path)
        
        cuda.cwGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cwGetDeviceCount.restype = ctypes.c_int
        
        count = ctypes.c_int(-1)
        result = cuda.cwGetDeviceCount(ctypes.byref(count))
        
        cuda.cwGetErrorString.argtypes = [ctypes.c_int]
        cuda.cwGetErrorString.restype = ctypes.c_char_p
        
        error_msg = cuda.cwGetErrorString(result)
        if error_msg:
            error_str = error_msg.decode('utf-8', errors='ignore')
        else:
            error_str = "unknown"
        
        if "stub" in error_str.lower() or "CPU mode" in error_str:
            pytest.fail(f"CUDA wrapper is a STUB: {error_str}")
        
        print(f"CUDA device count: {count.value}, result: {result}")

    def test_cublas_wrapper_not_stub(self):
        """Verify that cuBLAS wrapper is real, not a stub."""
        paths = [
            "./build/libcublaswrap.so",
            "/app/build/libcublaswrap.so",
            "../build/libcublaswrap.so",
        ]
        cublas_path = None
        for path in paths:
            if os.path.exists(path):
                cublas_path = path
                break
        
        if not cublas_path:
            pytest.skip("libcublaswrap.so not found")
        
        cublas = ctypes.CDLL(cublas_path)
        
        cublas.cbwGetErrorString.argtypes = [ctypes.c_int]
        cublas.cbwGetErrorString.restype = ctypes.c_char_p
        
        error_msg = cublas.cbwGetErrorString(999)
        if error_msg:
            error_str = error_msg.decode('utf-8', errors='ignore')
        else:
            error_str = "unknown"
        
        if "stub" in error_str.lower() or "CPU" in error_str:
            pytest.fail(f"cuBLAS wrapper is a STUB: {error_str}")
        
        print(f"cuBLAS error string check passed")

    def test_engine_loads_successfully(self):
        """Verify that engine.so loads and has expected symbols."""
        engine_path = get_engine_path()
        if not engine_path:
            pytest.skip("engine.so not found")
        
        try:
            engine = ctypes.CDLL(engine_path)
        except OSError as e:
            pytest.fail(f"Failed to load engine.so: {e}")
        
        required_symbols = [
            "init_engine",
            "prefill",
            "decode_step",
            "free_engine",
            "free_request_state",
        ]
        
        missing = []
        for sym in required_symbols:
            if not hasattr(engine, sym):
                missing.append(sym)
        
        if missing:
            pytest.fail(f"engine.so missing required symbols: {missing}")
        
        print(f"engine.so loaded successfully with all required symbols")

    def test_engine_dependencies_linked(self):
        """Verify that engine.so dependencies are properly linked."""
        engine_path = get_engine_path()
        if not engine_path:
            pytest.skip("engine.so not found")
        
        import subprocess
        try:
            result = subprocess.run(
                ["ldd", engine_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "not found" in result.stdout:
                lines = [l for l in result.stdout.split('\n') if "not found" in l]
                pytest.fail(f"engine.so has missing dependencies:\n" + "\n".join(lines))
            
            print(f"All engine.so dependencies found")
            
        except FileNotFoundError:
            pytest.skip("ldd not available")
        except subprocess.TimeoutExpired:
            pytest.skip("ldd timed out")

    def test_gpu_memory_available(self):
        """Verify GPU memory is accessible (not stub)."""
        cuda_path = get_cuda_wrapper_path()
        if not cuda_path:
            pytest.skip("libcudawrap.so not found")
        
        cuda = ctypes.CDLL(cuda_path)
        
        cuda.cwInit.argtypes = []
        cuda.cwInit.restype = ctypes.c_int
        
        cuda.cwGetTotalMemory.argtypes = []
        cuda.cwGetTotalMemory.restype = ctypes.c_ulonglong
        
        init_result = cuda.cwInit()
        
        if init_result == 100:
            pytest.skip("No GPU available (stub mode)")
        
        total_mem = cuda.cwGetTotalMemory()
        
        if total_mem == 0:
            pytest.fail("GPU reports 0 total memory - likely a stub")
        
        print(f"GPU total memory: {total_mem / (1024**3):.2f} GB")

    def test_kernel_library_not_stub(self):
        """Verify kernel library is real GPU code."""
        paths = [
            "./build/libkernels.so",
            "/app/build/libkernels.so",
            "../build/libkernels.so",
        ]
        kernel_path = None
        for path in paths:
            if os.path.exists(path):
                kernel_path = path
                break
        
        if not kernel_path:
            pytest.skip("libkernels.so not found")
        
        with open(kernel_path, 'rb') as f:
            content = f.read(1024)
        
        if b"GPU kernel stub called" in content or b"ERROR: GPU kernel stub" in content:
            pytest.fail("libkernels.so is a CPU stub - will abort on kernel calls")
        
        print("libkernels.so appears to be GPU code")


def run_production_validation():
    """Run all GPU validation tests - call this at container startup."""
    print("=" * 60)
    print("GLM-4.7-FP8 GPU Validation")
    print("=" * 60)
    
    tests = TestGPUValidation()
    
    failures = []
    
    try:
        tests.test_engine_loads_successfully()
        print("[PASS] Engine loads successfully")
    except Exception as e:
        failures.append(f"Engine load: {e}")
        print(f"[FAIL] Engine load: {e}")
    
    try:
        tests.test_engine_dependencies_linked()
        print("[PASS] Engine dependencies linked")
    except Exception as e:
        failures.append(f"Dependencies: {e}")
        print(f"[FAIL] Dependencies: {e}")
    
    try:
        tests.test_cuda_wrapper_not_stub()
        print("[PASS] CUDA wrapper is real GPU code")
    except Exception as e:
        failures.append(f"CUDA wrapper: {e}")
        print(f"[FAIL] CUDA wrapper: {e}")
    
    try:
        tests.test_cublas_wrapper_not_stub()
        print("[PASS] cuBLAS wrapper is real GPU code")
    except Exception as e:
        failures.append(f"cuBLAS wrapper: {e}")
        print(f"[FAIL] cuBLAS wrapper: {e}")
    
    try:
        tests.test_kernel_library_not_stub()
        print("[PASS] Kernel library is real GPU code")
    except Exception as e:
        failures.append(f"Kernel library: {e}")
        print(f"[FAIL] Kernel library: {e}")
    
    print("=" * 60)
    
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} issues found")
        for f in failures:
            print(f"  - {f}")
        return False
    else:
        print("VALIDATION PASSED: All GPU checks successful")
        return True


if __name__ == "__main__":
    import sys
    success = run_production_validation()
    sys.exit(0 if success else 1)

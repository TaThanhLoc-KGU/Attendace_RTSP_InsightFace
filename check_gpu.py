#!/usr/bin/env python3
"""
GPU/TensorRT Setup Diagnostic Script
"""
import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, capture=True):
    """Run command and return output"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            return subprocess.run(cmd, shell=True).returncode
    except Exception as e:
        return -1, "", str(e)


def check_nvidia_setup():
    """Check NVIDIA GPU and drivers"""
    print("=" * 60)
    print("1. NVIDIA GPU & DRIVER CHECK")
    print("=" * 60)

    # Check nvidia-smi
    code, stdout, stderr = run_command("nvidia-smi")
    if code == 0:
        print("✅ nvidia-smi working:")
        print(stdout[:500] + "..." if len(stdout) > 500 else stdout)

        # Extract CUDA version
        lines = stdout.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version: ')[-1].split()[0]
                print(f"🔍 Driver CUDA Version: {cuda_version}")
                break
    else:
        print("❌ nvidia-smi not working:")
        print(f"Error: {stderr}")
        return False

    # Check nvcc
    code, stdout, stderr = run_command("nvcc --version")
    if code == 0:
        print("✅ NVCC found:")
        print(stdout.strip())
    else:
        print("⚠️ NVCC not found (OK if using conda/pre-compiled)")

    return True


def check_cuda_installation():
    """Check CUDA installation"""
    print("\n" + "=" * 60)
    print("2. CUDA INSTALLATION CHECK")
    print("=" * 60)

    # Check CUDA paths
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Program Files\\NVIDIA Corporation\\NVIDIA GPU Computing Toolkit\\CUDA",
        os.environ.get('CUDA_PATH', '')
    ]

    found_cuda = False
    for path in cuda_paths:
        if path and Path(path).exists():
            print(f"✅ Found CUDA at: {path}")
            # List versions
            for version_dir in Path(path).glob("v*"):
                print(f"   Version: {version_dir.name}")
            found_cuda = True
            break

    if not found_cuda:
        print("❌ CUDA installation not found in standard locations")
        print("💡 Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")

    # Check environment variables
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'PATH']
    for var in cuda_vars:
        value = os.environ.get(var, '')
        if 'cuda' in value.lower() or var in ['CUDA_PATH', 'CUDA_HOME']:
            print(f"✅ {var}: {value[:100]}..." if len(value) > 100 else f"✅ {var}: {value}")

    return found_cuda


def check_cudnn():
    """Check cuDNN installation"""
    print("\n" + "=" * 60)
    print("3. cuDNN CHECK")
    print("=" * 60)

    # Check for cuDNN DLLs
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path:
        cudnn_dll_paths = [
            Path(cuda_path) / "bin" / "cudnn64_8.dll",
            Path(cuda_path) / "bin" / "cudnn_cnn_infer64_8.dll",
            Path(cuda_path) / "bin" / "cudnn_ops_infer64_8.dll"
        ]

        found_cudnn = False
        for dll_path in cudnn_dll_paths:
            if dll_path.exists():
                print(f"✅ Found cuDNN: {dll_path}")
                found_cudnn = True

        if not found_cudnn:
            print("❌ cuDNN DLLs not found")
            print("💡 Download cuDNN from: https://developer.nvidia.com/cudnn")

        return found_cudnn
    else:
        print("⚠️ Cannot check cuDNN - CUDA_PATH not set")
        return False


def check_tensorrt():
    """Check TensorRT installation"""
    print("\n" + "=" * 60)
    print("4. TensorRT CHECK")
    print("=" * 60)

    # Check Python TensorRT
    try:
        import tensorrt as trt
        print(f"✅ TensorRT Python: {trt.__version__}")

        # Test TensorRT functionality
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print("✅ TensorRT builder created successfully")
        return True

    except ImportError:
        print("❌ TensorRT Python package not found")
        print("💡 Install: pip install tensorrt")
        return False
    except Exception as e:
        print(f"❌ TensorRT error: {e}")
        return False


def check_onnxruntime():
    """Check ONNXRuntime installation"""
    print("\n" + "=" * 60)
    print("5. ONNXRUNTIME CHECK")
    print("=" * 60)

    # Check installed packages
    code, stdout, stderr = run_command("pip list | findstr onnxruntime")
    if code == 0:
        print("📦 Installed ONNX packages:")
        print(stdout)
    else:
        print("❌ No ONNX Runtime packages found")

    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"✅ ONNXRuntime version: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"🔍 Available providers: {providers}")

        # Test each provider
        for provider in providers:
            try:
                # Create a dummy session to test provider
                session = ort.InferenceSession(None, providers=[provider])
                print(f"✅ {provider}: Working")
            except Exception as e:
                print(f"❌ {provider}: {str(e)[:100]}")

        return True

    except Exception as e:
        print(f"❌ ONNXRuntime error: {e}")
        return False


def check_insightface():
    """Check InsightFace setup"""
    print("\n" + "=" * 60)
    print("6. INSIGHTFACE CHECK")
    print("=" * 60)

    try:
        import insightface
        print(f"✅ InsightFace version: {insightface.__version__}")

        from insightface.app import FaceAnalysis

        # Test with different providers
        provider_tests = [
            ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            ['CPUExecutionProvider']
        ]

        for i, providers in enumerate(provider_tests):
            try:
                print(f"\n🔍 Testing provider set {i + 1}: {providers}")
                app = FaceAnalysis(name='buffalo_s', providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 480))
                print(f"✅ Provider set {i + 1}: SUCCESS")

                # Performance test
                import numpy as np
                test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                import time
                start = time.time()
                faces = app.get(test_img)
                end = time.time()

                print(f"   Performance: {(end - start) * 1000:.1f}ms, {len(faces)} faces")
                break

            except Exception as e:
                print(f"❌ Provider set {i + 1}: {str(e)[:100]}")

        return True

    except Exception as e:
        print(f"❌ InsightFace error: {e}")
        return False


def generate_fix_commands():
    """Generate fix commands"""
    print("\n" + "=" * 60)
    print("7. RECOMMENDED FIX COMMANDS")
    print("=" * 60)

    print("🔧 Step 1: Uninstall conflicting packages")
    print("pip uninstall onnxruntime onnxruntime-gpu tensorrt -y")

    print("\n🔧 Step 2: Install GPU-enabled packages")
    print("pip install onnxruntime-gpu")
    print("pip install tensorrt")

    print("\n🔧 Step 3: Verify CUDA/cuDNN versions")
    print("Check compatibility matrix at:")
    print("https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html")

    print("\n🔧 Step 4: Set environment variables (if needed)")
    print("set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")
    print("set PATH=%CUDA_PATH%\\bin;%PATH%")

    print("\n🔧 Step 5: Test installation")
    print("python -c \"import onnxruntime as ort; print(ort.get_available_providers())\"")


def main():
    print("🚀 GPU/TensorRT SETUP DIAGNOSTIC")
    print("🎯 Diagnosing why ONNX Runtime falls back to CPU")
    print()

    checks = [
        check_nvidia_setup,
        check_cuda_installation,
        check_cudnn,
        check_tensorrt,
        check_onnxruntime,
        check_insightface
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)

    issues = []
    if not results[0]: issues.append("NVIDIA GPU/Driver issues")
    if not results[1]: issues.append("CUDA installation missing")
    if not results[2]: issues.append("cuDNN not found")
    if not results[3]: issues.append("TensorRT not installed")
    if not results[4]: issues.append("ONNXRuntime GPU support broken")
    if not results[5]: issues.append("InsightFace GPU not working")

    if not issues:
        print("✅ All checks passed! GPU should work.")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")

    # Generate fixes
    generate_fix_commands()


if __name__ == "__main__":
    main()
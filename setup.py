# setup.py
import os
import sys

def gen_compile_args_from_compute_cap(GPU_Compute_Capability_Major, GPU_Compute_Capability_Minor):
    """生成编译参数"""
    compile_dicts = {
        "sources": [],
        "extra_compile_args": {},
        "cuda_arch_v": 0
    }

    compile_dicts['cuda_arch_v'] = GPU_Compute_Capability_Major*100+GPU_Compute_Capability_Minor*10

    if 750 == compile_dicts['cuda_arch_v']: #Turing
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu"]
        compile_dicts['extra_compile_args'] = {
            'nvcc': [
                '-O3', 
                "-std=c++17",
                '--compiler-options', '-fPIC',
                '-gencode=arch=compute_75, code=sm_75',
            ]
        }
    elif 800 == compile_dicts['cuda_arch_v'] or 860 == compile_dicts['cuda_arch_v'] or 870 == compile_dicts['cuda_arch_v']: #Ampere
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu"] + [
            "csrc/gptq_marlin/"+filename for filename in os.listdir("csrc/gptq_marlin") if filename.split(".")[-1]=="cu"]
        compile_dicts['extra_compile_args'] = {
                                            'nvcc': [
                                                # "-O3",    # can't use for marlin 
                                                "-std=c++17",
                                                '--compiler-options', '-fPIC',
                                                '-gencode=arch=compute_80, code=sm_80',
                                                '-gencode=arch=compute_86, code=sm_86',
                                                '-gencode=arch=compute_87, code=sm_87'
                                            ]
        }
    elif 890 == compile_dicts['cuda_arch_v']: #Ada Lovelace
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu"]
        compile_dicts['extra_compile_args'] = {
            'nvcc': [
                '-DNDEBUG',
                '-O3', 
                "-std=c++17",
                '--compiler-options', '-fPIC',
                '-gencode=arch=compute_89, code=sm_89',
            ]
        }
    elif 900 == compile_dicts['cuda_arch_v']: #Hopper
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu"] + [
            "csrc/gptq_marlin/"+filename for filename in os.listdir("csrc/gptq_marlin") if filename.split(".")[-1]=="cu"]
        compile_dicts['extra_compile_args'] = {
                                        'nvcc': [
                                                "-DNDEBUG",
                                                # "-O3",    # can't use for marlin 
                                                "-Xcompiler",
                                                "-fPIC",
                                                # "-gencode=arch=compute_90,code=sm_90",
                                                "-gencode=arch=compute_90a,code=sm_90a",
                                                "-std=c++17",
                                        ]
            
        }
    elif 1100 == compile_dicts['cuda_arch_v']: #Blackwell
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu"] + [
            "csrc/gptq_marlin/"+filename for filename in os.listdir("csrc/gptq_marlin") if filename.split(".")[-1]=="cu"]
        compile_dicts['extra_compile_args'] = {
                                            'nvcc': [
                                                # "-O3",    # can't use for marlin 
                                                "-std=c++17",
                                                '--compiler-options', '-fPIC',
                                                '-gencode=arch=compute_110, code=sm_110',
                                            ]
        }
    else:
        sys.exit(f"No implemented for current compute capability: {GPU_Compute_Capability_Major}.{GPU_Compute_Capability_Minor}")

    return compile_dicts


try:
    import torch
except ImportError:
    sys.exit("Error: PyTorch not found. Please install PyTorch first.")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.cuda.is_available():
    sys.exit("Error: CUDA is not available.")

major, minor = torch.cuda.get_device_capability()
print(f"=========CUDA Compute Capability: {major}.{minor}==========")

cuda_lib_path = os.environ.get('CUDA_LIB_PATH', '/usr/local/cuda/lib64')
if not os.path.exists(cuda_lib_path):
    sys.exit(f"CUDA library path does not exist: {cuda_lib_path}")

compile_args = gen_compile_args_from_compute_cap(major, minor)

setup(
    ext_modules=[
        CUDAExtension(
            name='purevlm.cuda_ops',
            sources=compile_args['sources'],
            define_macros=[('HOST_CUDA_ARCH', compile_args['cuda_arch_v']),],
            extra_compile_args=compile_args['extra_compile_args'],
            include_dirs=[
                os.path.join(os.getcwd(), 'csrc/include'),
                os.path.join(os.getcwd(), 'csrc/gptq_marlin'),
                # os.path.join(os.getcwd(), 'csrc/include/cutlass/include'),
                # os.path.join(os.getcwd(), 'csrc/include/cutlass/tools/util/include'),
                # os.path.join(os.getcwd(), 'csrc/include/attention'),
            ],
            libraries=['cuda'],
            library_dirs=[cuda_lib_path],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
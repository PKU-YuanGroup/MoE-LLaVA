from torch.utils import cpp_extension
import pathlib
import os
import subprocess

def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")

# Check if cuda 11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=sm_80')
    if int(bare_metal_minor) >= 7:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_90,code=sm_90')

# Build path
srcpath = pathlib.Path(__file__).parent.absolute()
buildpath = srcpath / 'build'
_create_build_dir(buildpath)

def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
    return cpp_extension.load(
        name=name,
        sources=sources,
        build_directory=buildpath,
        extra_cflags=['-O3', ],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '--use_fast_math'] + extra_cuda_flags + cc_flag,
        verbose=1
    )

extra_flags = []

cache_autogptq_cuda_256_sources = ["./cache_autogptq_cuda_256.cpp",
           "./cache_autogptq_cuda_kernel_256.cu"]
cache_autogptq_cuda_256 = _cpp_extention_load_helper("cache_autogptq_cuda_256", cache_autogptq_cuda_256_sources, extra_flags)

"""CUDA 13 pip locks and compiler requirements for x86 and ARM wheels."""

CUDA13_X86_COMPILER = ["nvidia-cutlass-dsl==4.4.2", "nvidia-cutlass-dsl-libs-cu13==4.4.2"]
CUDA13_ARM_COMPILER = ["nvidia-cutlass-dsl==4.5.0", "nvidia-cutlass-dsl-libs-base==4.5.0"]

CUDA13_X86_LOCK = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda13.txt"
CUDA13_ARM_LOCK = "@rtp_pip_deps//:requirements_lock_cuda13_arm.txt"

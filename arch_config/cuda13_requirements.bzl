"""CUDA 13 compiler requirements embedded in the platform wheels."""

CUDA13_X86_COMPILER = ["nvidia-cutlass-dsl==4.4.2", "nvidia-cutlass-dsl-libs-cu13==4.4.2"]
CUDA13_ARM_COMPILER = ["nvidia-cutlass-dsl==4.5.0", "nvidia-cutlass-dsl-libs-base==4.5.0"]

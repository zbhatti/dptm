gcc src/clKernelTest.c -oKernelTest  -I/opt/intel/include -lOpenCL
gcc src/clTimingKernel.c -oTimingKernel  -I/opt/intel/include -lOpenCL
gcc src/clTimingKernelOptimal.c -oTimingKernelOptimal -I/opt/intel/include -lOpenCL
nvcc src/cudaTimingKernel.cu -o cudaTimingKernel


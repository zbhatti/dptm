gcc host.c -o Default  -I/opt/intel/include -lOpenCL
gcc -Wall host.c -o Intel -l:/opt/intel/opencl/lib64/libOpenCL.so.1.2



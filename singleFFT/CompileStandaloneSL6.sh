gcc -o ClientStandaloneCPU src/ClientStandaloneCPU.c -I/usr/include/ -lfftw3_threads -lfftw3 -lm -std=c99 -D _XOPEN_SOURCE
gcc -o ClientStandaloneOpenCL src/ClientStandaloneOpenCL.c -I/opt/intel/include -lOpenCL  -I/opt/clFFT/src/package/include  -L/opt/clFFT/src/package/lib64 -lclFFT -std=c99 -D _XOPEN_SOURCE
nvcc -o ClientStandaloneCUDA  src/ClientStandaloneCUDA.cu  -lcufft -lstdc++ -lm
gcc -o ClientStandaloneCPUThreadLoop src/ClientStandaloneCPUThreadLoop.c -I/usr/include -lfftw3_threads -lfftw3 -lm -std=c99 -D _XOPEN_SOURCE

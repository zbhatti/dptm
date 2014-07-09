gcc-mp-4.7 -o ClientStandaloneCPU src/ClientStandaloneCPU.c -I/usr/local/include -I/Users/zxb0111/Downloads/fftw-3.3.3/threads -lfftw3_threads -lfftw3 -lm -std=c99
gcc src/ClientStandaloneOpenCL.c -framework OpenCL -o ClientStandaloneOpenCL -I/Users/zxb0111/clFFT/src/include -L/Users/zxb0111/clFFT/src/library/ -lclFFT
nvcc -o ClientStandaloneCUDA  src/ClientStandaloneCUDA.cu  -lcufft -lstdc++ -lm
gcc-mp-4.7 -o ClientStandaloneCPUThreadLoop src/ClientStandaloneCPUThreadLoop.c -I/usr/local/include -I/Users/zxb0111/Downloads/fftw-3.3.3/threads -lfftw3_threads -lfftw3 -lm -std=c99

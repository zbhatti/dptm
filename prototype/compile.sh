nvcc -g -G -ccbin /opt/local/bin/gcc-mp-4.7 -o client client.cu -lstdc++ -I/usr/local/include/ -lfftw3 -lm -lcufft
nvcc -ccbin /opt/local/bin/gcc-mp-4.7 -o wrapper_FFT wrapper_FFT.cu -lstdc++ -I/usr/local/include/ -lm -lcufft
nvcc -ccbin /opt/local/bin/gcc-mp-4.7 -o DPTM DPTM.cu -lstdc++ -I/usr/local/include/ -lm -lcufft


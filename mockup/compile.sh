g++ \
src/bmpfuncs.c \
src/errorCheck.cpp \
src/dpKernel.cpp \
src/dpClient.cpp \
src/dpSquareArray.cpp \
src/dpRotateImage.cpp \
src/dpTiming.cpp \
src/dpMatrixMultiplication.cpp \
src/dpFFT.cpp \
-odpClient \
-I/opt/intel/include \
-I/home/zxb0111/dptm/mockup/include \
-lOpenCL \
-I/opt/clFFT/src/package/include \
-L/opt/clFFT/src/package/lib64 \
-lclFFT 


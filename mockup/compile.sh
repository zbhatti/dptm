g++ \
src/main.cpp \
src/dpClient.cpp \
src/dpKernelFactory.cpp \
src/bmpfuncs.cpp \
src/errorCheck.cpp \
src/dpSquareArray.cpp \
src/dpRotateImage.cpp \
src/dpTiming.cpp \
src/dpMatrixMultiplication.cpp \
src/dpFFT.cpp \
src/dpKernel.cpp \
-odpClient \
-I/opt/intel/include \
-I./include \
-lOpenCL \
-I/opt/clFFT/src/package/include \
-L/opt/clFFT/src/package/lib64 \
-lclFFT 


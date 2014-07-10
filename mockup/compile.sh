g++ \
helperFunctions/bmpfuncs.c \
helperFunctions/errorCheck.cpp \
dpKernel.cpp \
dpClient.cpp \
dpSquareArray.cpp \
dpMatrixMultiplication.cpp \
dpFFT.cpp \
-odpClient \
-I/opt/intel/include \
-lOpenCL \
-I/opt/clFFT/src/package/include \
-L/opt/clFFT/src/package/lib64 \
-lclFFT 


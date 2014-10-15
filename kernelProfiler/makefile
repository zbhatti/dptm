CFLAGS=-c -Wall -Wno-comment -Wno-write-strings -g
NVCCFLAGS=-c -g --compiler-options -Wno-write-strings 
INCCUDA=-I./include -I/opt/AMDAPP/include -I/usr/local/cuda/include
IDIR=-I./include -I/opt/AMDAPP/include -I/usr/local/cuda/include
LIBDIR= -L/usr/lib64  -L/opt/clFFT/src/package/lib64 -L/usr/local/cuda/lib64
LDFLAGS=  -lOpenCL -lcudart
#exclude main files due to having multiple main definitions at linking time:
#cuda .cu files are replaced with .cpp for consistency in SRCLIST variable
SRCLIST= \
errorCheck.cpp \
dpKernel.cpp \
dpClient.cpp \
dpSquareArray.cpp \
dpMatrixMultiplication.cpp \
dpConvolution.cpp \
dpComputation.cpp \
dpMemory.cpp \
dpMatrixTranspose.cpp \
dpOxxxxx.cpp \
dpVectorAdd.cpp \
dpLUDecomposition.cpp \
dpNBody.cpp \
dpEmpty.cpp \
dpNoMemory.cpp \
dpFWT.cpp \
dpFFT.cpp \
dpCudaEmpty.cpp \
dpCudaNoMemory.cpp \
dpCudaComputation.cpp \
dpCudaMemory.cpp \
dpCudaFFT.cpp \
dpCudaSquareArray.cpp \
dpCudaVectorAdd.cpp \
dpCudaMatrixTranspose.cpp \
dpCudaMatrixMultiplication.cpp \
dpFloydWarshall.cpp \
dpFluidSimulation.cpp \
dpArray3dAverage.cpp \
dpReduction.cpp \
dpMonteCarloAsian.cpp \
dpTiming.cpp \
dpKernelFactory.cpp \
bmpfuncs.cpp
SOURCES=$(addprefix src/, $(SRCLIST))
OBJECTS=$(addprefix obj/, $(SRCLIST))

all: main

main: dpClientSrc dpClientShell

dpClientSrc: $(OBJECTS:.cpp=.o)
	g++ $(CFLAGS) $(INCCUDA) -o obj/mainSrc.o src/mainSrc.cpp
	g++ $(OBJECTS:.cpp=.o) obj/mainSrc.o -g -std=c++11 -o $@ $(LIBDIR) $(LDFLAGS) $(IDIR)

dpClientShell: $(OBJECTS:.cpp=.o)
	g++ $(CFLAGS) $(INCCUDA) -o obj/mainShell.o src/mainShell.cpp 
	g++ $(OBJECTS:.cpp=.o) obj/mainShell.o -g -std=c++11 -o $@ $(LIBDIR) $(LDFLAGS) $(IDIR)


#EXECUTION:
obj/dpClient.o: src/dpClient.cpp
	g++ $(CFLAGS) $(INCCUDA) -o $@ $<

obj/dpKernel.o: src/dpKernel.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $<

obj/dpKernelFactory.o: src/dpKernelFactory.cpp
	g++ $(CFLAGS) $(IDIR) -I/opt/clFFT/src/include -o $@ $< 
	
##KERNELS:
obj/dpArray3dAverage.o: src/dpArray3dAverage.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpConvolution.o: src/dpConvolution.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 	

obj/dpComputation.o: src/dpComputation.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 	

obj/dpEmpty.o: src/dpEmpty.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpNoMemory.o: src/dpNoMemory.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 	
	
obj/dpFFT.o: src/dpFFT.cpp
	g++ $(CFLAGS) -I./include -I/opt/AMDAPP/include -I/opt/clFFT/src/include -o $@ $< 

obj/dpFloydWarshall.o: src/dpFloydWarshall.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpFWT.o: src/dpFWT.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpFluidSimulation.o: src/dpFluidSimulation.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpLUDecomposition.o: src/dpLUDecomposition.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpMatrixMultiplication.o: src/dpMatrixMultiplication.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/dpMatrixTranspose.o: src/dpMatrixTranspose.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

obj/dpMemory.o: src/dpMemory.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 	
	
obj/dpMonteCarloAsian.o: src/dpMonteCarloAsian.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

obj/dpNBody.o: src/dpNBody.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

obj/dpOxxxxx.o: src/dpOxxxxx.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

obj/dpReduction.o: src/dpReduction.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 	
	
obj/dpSquareArray.o: src/dpSquareArray.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

obj/dpVectorAdd.o: src/dpVectorAdd.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

#CUDA KERNELS:
obj/dpCudaFFT.o: src/dpCudaFFT.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $<
	
obj/dpCudaSquareArray.o: src/dpCudaSquareArray.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $<

obj/dpCudaVectorAdd.o: src/dpCudaVectorAdd.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $<

obj/dpCudaMatrixMultiplication.o: src/dpCudaMatrixMultiplication.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $<

obj/dpCudaMatrixTranspose.o: src/dpCudaMatrixTranspose.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $< 
	
obj/dpCudaEmpty.o: src/dpCudaEmpty.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $< 
	
obj/dpCudaNoMemory.o: src/dpCudaNoMemory.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $< 

obj/dpCudaMemory.o: src/dpCudaMemory.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $< 
	
obj/dpCudaComputation.o: src/dpCudaComputation.cu
	nvcc $(NVCCFLAGS) $(INCCUDA) -o $@ $< 
	
#HELPER FUNCTIONS:
obj/errorCheck.o: src/errorCheck.cpp
	g++ $(CFLAGS) $(INCCUDA) -o $@ $<

obj/dpTiming.o: src/dpTiming.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 
	
obj/bmpfuncs.o: src/bmpfuncs.cpp
	g++ $(CFLAGS) $(IDIR) -o $@ $< 

clean:
	rm dpClient* clfft.kernel*

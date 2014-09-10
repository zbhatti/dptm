CFLAGS=-c -Wall -g
includedir=-I/opt/clFFT/src/package/include -I./include -I/opt/AMDAPP/include
#includedir=-I/opt/clFFT/src/package/include -I./include -I/usr/local/cuda/include
libdir=-L/opt/clFFT/src/package/lib64
LDFLAGS=-lclFFT -lOpenCL
SRCLIST= \
errorCheck.cpp \
dpKernel.cpp \
dpClient.cpp \
dpSquareArray.cpp \
dpRotateImage.cpp \
dpMatrixMultiplication.cpp \
dpConvolution.cpp \
dpMersenneTwister.cpp \
dpMatrixTranspose.cpp \
dpVectorAdd.cpp \
dpLUDecomposition.cpp \
dpNBody.cpp \
dpFWT.cpp \
dpFFT.cpp \
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

main: objects
	g++ $(OBJECTS:.cpp=.o) obj/mainSrc.o -g -std=c++11 -o dpClientSrc $(includedir) $(libdir) $(LDFLAGS)
	g++ $(OBJECTS:.cpp=.o) obj/mainShell.o -g -std=c++11 -o dpClientShell $(includedir) $(libdir) $(LDFLAGS)

objects: $(SOURCES)
	gcc $(CFLAGS) $(includedir) $(libdir) $(LDFLAGS) $(SOURCES) src/mainSrc.cpp
	gcc $(CFLAGS) $(includedir) $(libdir) $(LDFLAGS) $(SOURCES) src/mainShell.cpp
	mv *.o obj/

#clean:
	#rm -rf *.o dpClient*
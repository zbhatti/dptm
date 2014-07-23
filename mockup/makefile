CFLAGS=-c -Wall 
includedir=-I/opt/clFFT/src/package/include -I./include -I/opt/intel/include
libdir=-L/opt/clFFT/src/package/lib64
LDFLAGS=-lclFFT -lOpenCL
SRCLIST= \
main.cpp \
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
dpTiming.cpp \
dpKernelFactory.cpp \
bmpfuncs.cpp
SOURCES=$(addprefix src/, $(SRCLIST))
OBJECTS=$(addprefix obj/, $(SRCLIST))

all: main

main: objects
	g++ $(OBJECTS:.cpp=.o) -std=c++11 -o dpClient $(includedir) $(libdir) $(LDFLAGS)

objects: $(SOURCES)
	gcc $(CFLAGS) $(includedir) $(libdir) $(LDFLAGS) $(SOURCES)
	mv *.o obj/

clean:
	rm -rf *o dpClient
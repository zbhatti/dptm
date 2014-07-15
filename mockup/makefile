CFLAGS=-c -Wall
includedir=-I/opt/clFFT/src/package/include -I/home/zxb0111/dptm/mockup/include -I/opt/intel/include
libdir=-L/opt/clFFT/src/package/lib64
LDFLAGS=-lclFFT -lOpenCL
SRCLIST= \
errorCheck.cpp \
dpKernel.cpp \
dpClient.cpp \
dpSquareArray.cpp \
dpRotateImage.cpp \
dpMatrixMultiplication.cpp \
dpFFT.cpp \
dpTiming.cpp \
bmpfuncs.cpp
SOURCES=$(addprefix src/, $(SRCLIST))
OBJECTS=$(addprefix obj/, $(SRCLIST))

all: main

main: $(OBJECTS)
	g++ $(OBJECTS:.cpp=.o) -o dpClient $(includedir) $(libdir) $(LDFLAGS)

$(OBJECTS): $(SOURCES)
	gcc $(CFLAGS) $(includedir) $(libdir) $(LDFLAGS) $(SOURCES)
	mv *.o obj/

clean:
	rm -rf *o dpClient
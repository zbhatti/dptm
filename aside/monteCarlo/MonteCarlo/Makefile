HEADERS=cmdoptions.hpp ../common/basic.hpp ../common/cmdparser.hpp ../common/oclobject.hpp
SOURCES=cmdoptions.cpp montecarlo.cpp ../common/basic.cpp ../common/cmdparser.cpp ../common/oclobject.cpp

all: montecarlo

montecarlo: $(HEADERS) $(SOURCES) Makefile
	g++ $(SOURCES) -I../common -lOpenCL -omontecarlo

clean:
	rm -f montecarlo


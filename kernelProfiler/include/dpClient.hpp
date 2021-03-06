#include <stdio.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <vector>
#include <math.h>
#include <string>
#include <string.h>
#include <fstream> //file writing
#include <iostream> //file writing
#include <sys/stat.h> //mkdir 
#include <sys/types.h> //mkdir
#include <sstream>
#include <algorithm>

#include <cuda.h>//cuda functions
#include <cuda_runtime.h>//cuda functions
#include <cufft.h>


#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include "enumeratedTypes.h"
#include "dpTiming.hpp"
#include "dpKernelFactory.hpp"

class dpClient {
	private:
		int device;
		int platform;
		cl_context context;
		cl_command_queue queue;
		size_t MaxWorkGroupSize;
		size_t MaxWorkDim[3];
		cl_ulong MaxMemAlloc;
		int MaxComputeUnits;
		struct timeval start, finish;
		std::vector<dpKernel*> taskList;
		std::vector<dpTiming> timeList;
		dpKernelFactory kernelFactory;
		char devName[256];
		char platName[256];
		char type[256];
		bool isEmpty(std::ifstream&);
		//void getOptimalWG(std::string, int);
		//void getOptimalMB(std::string, int, int, int);
	
	public:
		dpClient(int, int);
		float timeDiff(struct timeval, struct timeval);
		void addTask(std::string, int xLocal=1, int yLocal=1, int zLocal=1, int MB=8);
		void addMBScan(std::string, int xLocal=1 , int yLocal=1, int zLocal=1); //need to change this to lookup optimal
		void addWGScan(std::string, int MB = 8);
		void runTasks();
		void printScreen();
		void printFile();
		char* getDev(){return devName;};
		char* getPlat(){return platName;};
		char* getType(){return type;};
		std::vector<dpTiming> getTimes(){return timeList;};
};

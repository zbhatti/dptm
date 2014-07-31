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
		cl_platform_id platform_ids[16];
		cl_device_id device_ids[16];
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kernel;
		size_t MaxWorkGroupSize;
		size_t MaxWorkDim[3];
		int MaxComputeUnits;
		struct timeval start, finish;
		std::vector<dpKernel*> taskList;
		std::vector<dpTiming> timeList;
		dpKernelFactory kernelFactory;
		char name[256];
		
	
	public:
		dpClient(int, int);
		float timeDiff(struct timeval, struct timeval);
		void runTasks();
		void addTask(std::string, int);
		void addTask(std::string, int, int);
		void addTask(std::string, int, int, int);
		void addTaskScan(std::string);
		void runTaskScan(std::string);
		void printTimes();
		void printFile();
		bool isEmpty(std::ifstream&);
		std::vector<dpTiming> getTimes(){return timeList;};
};
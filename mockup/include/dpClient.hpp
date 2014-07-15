#include <stdio.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <vector>
#include <math.h>
#include <string>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include "enumeratedTypes.h"
#include "dpFFT.hpp"
#include "dpSquareArray.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpRotateImage.hpp"
#include "dpTiming.hpp"

class dpClient {
	private:
		cl_platform_id platform_ids[16];
		cl_device_id device_ids[16];
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kernel;
		char * kernelString;
		size_t MaxWorkGroupSize;
		int MaxComputeUnits;
		struct timeval start, finish;
		std::vector<dpKernel*> taskList;
		std::vector<dpTiming> timeList;
	
	public:
		dpClient(int,int);
		float timeDiff(struct timeval, struct timeval);
		void runKernels();
		void printTimes();
};
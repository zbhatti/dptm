#ifndef __dpKernel_H_INCLUDED__
#define __dpKernel_H_INCLUDED__
#include <stdio.h>
#include <stdlib.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <math.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include "helperFunctions/bmpfuncs.h"

class dpKernel{
	protected:
		cl_context context;
		cl_command_queue queue;
		cl_kernel kernel;
		cl_program program;
		size_t localSize[3], globalSize[3];
	public:
		void FillerFunction();
		
};

#endif
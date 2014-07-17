#ifndef __dpKernel_H_INCLUDED__
#define __dpKernel_H_INCLUDED__
#include <stdio.h>
#include <stdlib.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <math.h>
#include <string>
#include <vector>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include "enumeratedTypes.h"


class dpKernel{
	protected:
		cl_context context;
		cl_command_queue queue;
		cl_kernel kernel;
		cl_program program;
		size_t localSize[3];
		size_t globalSize[3];
		
	public:
		std::string name;
		std::vector<std::string> dataNames;
		std::vector<float> dataParameters;
		workGroupSpace workDimension;
		void FillerFunction();
		virtual void init(int,int,int) = 0;
		virtual void memoryCopyOut(void) = 0;
		virtual void plan(void) = 0;
		virtual void execute(void) = 0;
		virtual void memoryCopyIn(void) = 0;
		virtual void cleanUp(void) = 0;
		size_t* getLocalSize(){return localSize;};
};

#endif
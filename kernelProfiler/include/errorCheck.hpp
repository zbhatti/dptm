//http://tom.scogland.com/blog/2013/03/29/opencl-errors/
//http://stackoverflow.com/a/14038590/2573580


#include <stdio.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

const char* get_error_string(cl_int);
void clAssert(cl_int, const char*, int);
void cudaAssert(cudaError code, const char *file, int line);
void programCheck(cl_int, cl_context, cl_program);


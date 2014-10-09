#include "dpEmpty.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpEmpty::dpEmpty(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	
	//Empty Kernel
	kernelString = "__kernel void Empty(){}";
	name = "Empty";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err);
	clErrChk(err);
	
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "Empty", &err);
	clErrChk(err);
}

void dpEmpty::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	MB=dataMB;
}

void dpEmpty::init(){}

void dpEmpty::memoryCopyOut(){}

void dpEmpty::plan(){
	// Set the Argument values
	globalSize[0] = 512;
}

int dpEmpty::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clFinish(queue);
	return 0;
}

void dpEmpty::memoryCopyIn(){}

void dpEmpty::cleanUp(){
	// Cleanup and leave
	clReleaseKernel(kernel);  
	clReleaseProgram(program);
}

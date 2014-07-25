/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#include "dpFWT.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpFWT::dpFWT(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	name = "FWT";
	kernelString = "\n"
	"__kernel                                                             \n"
	"void fastWalshTransform(__global float * tArray,                     \n"
	"                        __const  int   step                          \n"
	"                       )                                             \n"
	"{                                                                    \n"
	"		unsigned int tid = get_global_id(0);                              \n"
	"		                                                                  \n"
	"        const unsigned int group = tid%step;                         \n"
	"        const unsigned int pair  = 2*step*(tid/step) + group;        \n"
	"                                                                     \n"
	"        const unsigned int match = pair + step;                      \n"
	"                                                                     \n"
	"        float T1          = tArray[pair];                            \n"
	"        float T2          = tArray[match];                           \n"
	"                                                                     \n"
	"        tArray[pair]             = T1 + T2;                          \n"
	"        tArray[match]            = T1 - T2;                          \n"
	"}                                                                    \n";
}

void dpFWT::init(int xLocal,int yLocal, int zLocal){
	localSize[0]= xLocal;
	localSize[1]= yLocal;
	localSize[2]= zLocal;
	
	length = 32768;
	if(length < 512)
		length = 512;
	
	dataParameters.push_back(length);
	dataNames.push_back("nElements");

	// allocate and init memory used by host
	input = (cl_float *) malloc(length * sizeof(cl_float));
	output = (cl_float *) malloc(length * sizeof(cl_float));
	generateArray(input, length);
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "fastWalshTransform", &err); clErrChk(err);

	
	
}

void dpFWT::memoryCopyOut(){
	inputBuffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length,0,&err); clErrChk(err);
	
	// Enqueue write input to inputBuffer
	clErrChk(clEnqueueWriteBuffer(queue,inputBuffer,CL_FALSE,0,length * sizeof(cl_float),input,0,NULL,NULL));
	clErrChk(clFinish(queue));
}

void dpFWT::plan(){
	
	/*
	* The kernel performs a butterfly operation and it runs for half the
	* total number of input elements in the array.
	* In each pass of the kernel two corresponding elements are found using
	* the butterfly operation on an array of numbers and their sum and difference
	* is stored in the same locations as the numbers
	*/
	globalSize[0] = length / 2;
	clErrChk(clSetKernelArg(kernel,0,sizeof(cl_mem),(void *)&inputBuffer));
}

void dpFWT::execute(){
	for(cl_int step = 1; step < length; step <<= 1){
		// stage of the algorithm
		clErrChk(clSetKernelArg(kernel,1,sizeof(cl_int),(void *)&step));
		clErrChk(clEnqueueNDRangeKernel(queue,kernel,1,NULL,globalSize,localSize,0,NULL,NULL));
	}
	clErrChk(clFinish(queue));	
}

void dpFWT::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue,inputBuffer,CL_FALSE,0,length * sizeof(cl_float),output,0,NULL,NULL));
	clErrChk(clFinish(queue));
}

void dpFWT::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(inputBuffer));

	// release program resources (input memory etc.)
	free(input);
	free(output);
}

void dpFWT::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}
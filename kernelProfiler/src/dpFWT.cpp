/**********************************************************************
Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.
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
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); 
	clErrChk(err);
	
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	programCheck(err, context, program);
	
	kernel = clCreateKernel(program, "fastWalshTransform", &err); 
	clErrChk(err);	
	
}

void dpFWT::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0]= xLocal;
	localSize[1]= 1;
	localSize[2]= 1;
	
	for (int i = 0; pow(2,i)*sizeof(cl_float)/(float) 1048576 < dataMB; i++)
		length = pow(2,i);
	
	//length = dataMB*1048576/sizeof(cl_float);
	
	if(length < 512)
		length = 512;
	
	//fprintf(stderr,"length: %d\n", length);
	MB = length*sizeof(cl_float)/1048576;
	
}


void dpFWT::init(){
	dataParameters.push_back(length);
	dataNames.push_back("nElements");

	// allocate and init memory used by host
	input = (cl_float *) malloc(length * sizeof(cl_float));
	output = (cl_float *) malloc(length * sizeof(cl_float));
	generateArray(input, length);
	
}

void dpFWT::memoryCopyOut(){
	inputBuffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_float) * length,0,&err); 
	clErrChk(err);
	
	// Enqueue write input to inputBuffer
	clErrChk(clEnqueueWriteBuffer(queue,inputBuffer,CL_TRUE,0,length * sizeof(cl_float),input,0,NULL,NULL));
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
	//fprintf(stderr, "%d,%d\n", globalSize[0],localSize[0]);
	clErrChk(clSetKernelArg(kernel,0,sizeof(cl_mem),(void *)&inputBuffer));
}

int dpFWT::execute(){
	int flag=0;
	int step;
	for(step = 1; step < length; step <<= 1){
		//printf("step: %d\n",step);
		// stage of the algorithm
		clErrChk(clSetKernelArg(kernel,1,sizeof(int),(void *)&step));
		err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,globalSize,localSize,0,NULL,NULL);
		clErrChk(err);
		if(err<0)
			flag = -1;
	}
	
	if (flag == -1)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpFWT::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue,inputBuffer,CL_TRUE,0,length * sizeof(cl_float),output,0,NULL,NULL));
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
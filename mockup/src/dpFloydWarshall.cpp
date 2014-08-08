/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#include "dpFloydWarshall.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpFloydWarshall::dpFloydWarshall(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;

	name = "FloydWarshall";
	kernelString = "\n"
	"inline unsigned int uintMin(unsigned int a, unsigned int b){                 \n"
	"    return (b < a) ? b : a;                                                  \n"
	"}                                                                            \n"
	"                                                                             \n"
	"__kernel void floydWarshallPass(__global uint * pathDistanceBuffer,          \n"
	"																__global uint * pathBuffer,                   \n"
	"																const unsigned int numNodes,                  \n"
	"																const unsigned int pass){                     \n"
	"    int xValue = get_global_id(0);                                           \n"
	"    int yValue = get_global_id(1);                                           \n"
	"                                                                             \n"
	"    int k = pass;                                                            \n"
	"    int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];          \n"
	"    int tempWeight = (pathDistanceBuffer[yValue * numNodes + k] +            \n"
	"			pathDistanceBuffer[k * numNodes + xValue]);                             \n"
	"                                                                             \n"
	"    if (tempWeight < oldWeight)                                              \n"
	"    {                                                                        \n"
	"        pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;         \n"
	"        pathBuffer[yValue * numNodes + xValue] = k;                          \n"
	"    }                                                                        \n"
	"}                                                                            \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "floydWarshallPass", &err); clErrChk(err);
	
}

void dpFloydWarshall::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = 1;
	
	for (int i =0; pow(2,i)*pow(2,i)*sizeof(cl_uint)/(float) 1048576 <= dataMB;i++){
		numNodes = pow(2,i);
	}
	
	MB = numNodes*numNodes*sizeof(cl_uint)/(float) 1048576;
}

void dpFloydWarshall::init(){
	dataParameters.push_back(numNodes);
	dataParameters.push_back(numNodes);
	dataNames.push_back("width");
	dataNames.push_back("height");
	
	cl_uint matrixSizeBytes = numNodes * numNodes * sizeof(cl_uint);
	pathDistanceMatrix = (cl_uint *) malloc(matrixSizeBytes);
	pathMatrix = (cl_uint *) malloc(matrixSizeBytes);

	generateMatrix(pathDistanceMatrix, numNodes, numNodes);
	for(cl_int i = 0; i < numNodes; ++i){
		cl_uint iXWidth = i * numNodes;
		pathDistanceMatrix[iXWidth + i] = 0;
	}

	for(cl_int i = 0; i < numNodes; ++i){
		for(cl_int j = 0; j < i; ++j){
			pathMatrix[i * numNodes + j] = i;
			pathMatrix[j * numNodes + i] = j;
		}
		pathMatrix[i * numNodes + i] = i;
	}

}
void dpFloydWarshall::memoryCopyOut(){
	pathDistanceBuffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_uint) * numNodes * numNodes,NULL,&err); clErrChk(err);
	pathBuffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,sizeof(cl_uint) * numNodes * numNodes,NULL, &err); clErrChk(err);
	
	clErrChk(clEnqueueWriteBuffer(queue,pathDistanceBuffer,CL_FALSE,0, sizeof(cl_uint) * numNodes * numNodes,
																pathDistanceMatrix,0,NULL,NULL));
	clFinish(queue);
}

void dpFloydWarshall::plan(){
	clErrChk(clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&pathDistanceBuffer));
	clErrChk(clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&pathBuffer));
	clErrChk(clSetKernelArg(kernel,2,sizeof(cl_uint),(void*)&numNodes));
	clErrChk(clSetKernelArg(kernel,3,sizeof(cl_uint),(void*)&numNodes));
	globalSize[0]=numNodes;
	globalSize[1]=numNodes;
	
}

int dpFloydWarshall::execute(){
	cl_uint numPasses = numNodes;
	for(cl_uint i = 0; i < numPasses; i += 1){
		/*
		* Kernel needs which pass of the algorithm is running
		* which is sent as the Fourth argument
		*/
		clErrChk(clSetKernelArg(kernel,3,sizeof(cl_uint),(void*)&i));
		// Enqueue a kernel run call.
		err= clEnqueueNDRangeKernel(queue,kernel,2,NULL,globalSize,localSize,0,NULL,NULL);
		clErrChk(err)
		if (err < 0)
			return -1;
		clFinish(queue);
	}
	clFinish(queue);
	return 0;
}

void dpFloydWarshall::memoryCopyIn(){
    clErrChk(clEnqueueReadBuffer(queue,pathBuffer,CL_TRUE,0,numNodes * numNodes * sizeof(cl_uint),
                                 pathMatrix,0,NULL,NULL));
																 
		clFinish(queue);
    clErrChk(clEnqueueReadBuffer(	queue,pathDistanceBuffer,CL_TRUE,0, numNodes * numNodes * sizeof(cl_uint),
																	pathDistanceMatrix,0,NULL,NULL));
		clFinish(queue);
}

void dpFloydWarshall::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(pathDistanceBuffer));
	clErrChk(clReleaseMemObject(pathBuffer));

	// release program resources (input memory etc.)
	free(pathDistanceMatrix);
	free(pathMatrix);
}

void dpFloydWarshall::generateMatrix(cl_uint *A, int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/1000000 + 1);
		}
	}
}

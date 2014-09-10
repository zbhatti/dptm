
/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "dpReduction.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpReduction::dpReduction(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;

	name = "Reduction";
	kernelString = "\n"
	"__kernel                                                                              \n"
	"void                                                                                  \n"
	"reduce(__global uint4* input, __global uint4* output, __local uint4* sdata)           \n"
	"{                                                                                     \n"
	"    // load shared mem                                                                \n"
	"    unsigned int tid = get_local_id(0);                                               \n"
	"    unsigned int bid = get_group_id(0);                                               \n"
	"    unsigned int gid = get_global_id(0);                                              \n"
	"                                                                                      \n"
	"    unsigned int localSize = get_local_size(0);                                       \n"
	"    unsigned int stride = gid * 2;                                                    \n"
	"    sdata[tid] = input[stride] + input[stride + 1];                                   \n"
	"                                                                                      \n"
	"    barrier(CLK_LOCAL_MEM_FENCE);                                                     \n"
	"    // do reduction in shared mem                                                     \n"
	"    for(unsigned int s = localSize >> 1; s > 0; s >>= 1)                              \n"
	"    {                                                                                 \n"
	"        if(tid < s)                                                                   \n"
	"        {                                                                             \n"
	"            sdata[tid] += sdata[tid + s];                                             \n"
	"        }                                                                             \n"
	"        barrier(CLK_LOCAL_MEM_FENCE);                                                 \n"
	"    }                                                                                 \n"
	"                                                                                      \n"
	"    // write result for this block to global mem                                      \n"
	"    if(tid == 0) output[bid] = sdata[0];                                              \n"
	"}                                                                                     \n";

	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clErrChk(err);
	programCheck(err, context, program);
	kernel = clCreateKernel(program, "reduce", &err); clErrChk(err);
}

void dpReduction::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
		
	length = 1048576*dataMB/(sizeof(cl_uint)*VECTOR_SIZE);
	
	if(length%(localSize[0]*VECTOR_SIZE)!= 0){
		length = length - length%(localSize[0]*VECTOR_SIZE) + localSize[0]*VECTOR_SIZE;
	}
	
	MB =  sizeof(cl_uint)*length*VECTOR_SIZE / 1048576;
}

void dpReduction::init(){
	
	input = (cl_uint*)memalign(16, length * sizeof(cl_uint4));

  // random initialisation of input
  fillRandom<cl_uint>(input, length * VECTOR_SIZE, 1, 0, 5);
	
	dataParameters.push_back(length);
	dataNames.push_back("length");
	
	/* Allocate memory for output buffer as output depends on groupSize
	which also depends on device */
	numBlocks = length / ((cl_uint)localSize[0] * MULTIPLY);
	outputPtr = (cl_uint*)malloc(numBlocks * VECTOR_SIZE * sizeof(cl_uint));
	memset(outputPtr, 0, numBlocks * VECTOR_SIZE * sizeof(cl_uint));
	
	
}

void dpReduction::memoryCopyOut(){

	inputBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY,length * sizeof(cl_uint4),NULL,&err);
	// Create memory objects for temporary output array
	outputBuffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,numBlocks * sizeof(cl_uint4),NULL,&err);
	clErrChk(err);
	clErrChk(err);
	clErrChk(clFinish(queue));
	
	// Arguments are set and execution call is enqueued on command buffer
	// Transfer input to device
	void* mapPtr = clEnqueueMapBuffer(queue,inputBuffer,CL_FALSE,CL_MAP_WRITE,0,
																		length * sizeof(cl_uint4),0,NULL,NULL,&err);

	memcpy(mapPtr, input, length * sizeof(cl_uint4));
	clErrChk(clEnqueueUnmapMemObject(queue,inputBuffer,mapPtr,0,NULL,NULL));
	
	
}

void dpReduction::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outputBuffer));
	clErrChk(clSetKernelArg(kernel, 2, localSize[0] * sizeof(cl_uint4), NULL));
	globalSize[0] = length / MULTIPLY;
}

int dpReduction::execute(){

	// Enqueue a kernel run call.
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL); 
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpReduction::memoryCopyIn(){
	 cl_uint* outMapPtr = (cl_uint*)clEnqueueMapBuffer(queue,outputBuffer,CL_FALSE,CL_MAP_READ,0,
                         numBlocks * sizeof(cl_uint4),0,NULL,NULL,&err); 
	clErrChk(err);	
	clErrChk(clFlush(queue));
		
	// Add individual sum of blocks
  output = 0;
	for(int i = 0; i < numBlocks * VECTOR_SIZE; ++i)
		output += outMapPtr[i];
	
	
	clErrChk(clEnqueueUnmapMemObject(queue,outputBuffer,(void*)outMapPtr,0,NULL,NULL));
	clErrChk(clFinish(queue));
}

void dpReduction::cleanUp(){
	
	// Releases OpenCL resources (Memory etc.)
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(inputBuffer));
	clErrChk(clReleaseMemObject(outputBuffer));

	// release program resources (input memory etc.)
	free(input);
	free(outputPtr);

}

/**
 * fillRandom (FROM SDKUtil.hpp)
 * fill array with random values
 */
template<typename T> int dpReduction::fillRandom(T * arrayPtr,const int width,const int height,const T rangeMin,const T rangeMax){
	srand(time(NULL));
	double range = double(rangeMax - rangeMin) + 1.0;
	/* random initialisation of input */
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			int index = i*width + j;
			arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
		}
	}
	return 0;
}

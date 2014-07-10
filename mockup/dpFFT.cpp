#include "dpFFT.hpp"
#include "helperFunctions/errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }


dpFFT::dpFFT(float* input, int inputSize, cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	clErrChk(clfftSetup(&fftSetup));
	Ain = input;
	Asize = inputSize;
	Aout = new float[Asize*2];
	if (!Aout)
		fprintf(stderr,"error in dynamic allocation");
	
}

void dpFFT::memoryCopyOut(){
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpFFT::plan(){
	clLengths[0] = (size_t) Asize;
	clErrChk(clfftCreateDefaultPlan(&planHandle, context, CLFFT_1D, clLengths));
	clErrChk(clfftSetPlanPrecision(planHandle, CLFFT_SINGLE));
	clErrChk(clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
	clErrChk(clfftSetResultLocation(planHandle, CLFFT_INPLACE));
	clErrChk(clfftBakePlan(planHandle, 1, &queue, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpFFT::execute(){
	clErrChk(clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL));
	clErrChk(clFinish(queue));
}

void dpFFT::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpFFT::cleanUp(){
	clErrChk(clReleaseMemObject(buffer));
	delete[] Aout;
	clErrChk(clfftDestroyPlan(&planHandle));
	clfftTeardown();
}
/*
#include "dpFFT.hpp"
#include "errorCheck.hpp"
#include <clFFT.h>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpFFT::dpFFT(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	name = "FFT";
	
	clErrChk(clfftSetup(&fftSetup));

}

void dpFFT::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] =localSize[1]=localSize[2]=1;
	
	Asize = (dataMB*1048576)/(2*sizeof(float));
	if (Asize%2 != 0)
		Asize = Asize + 1;
		
	MB = Asize*2*sizeof(float)/1048576;

}

void dpFFT::init(){

	dataParameters.push_back(Asize);
	dataNames.push_back("nVectors");

	Ain = (float*) malloc( sizeof(float)*Asize*2);
	Aout = (float*) malloc( sizeof(float)*Asize*2);
	if (!Aout || !Ain)
		fprintf(stderr,"error in allocation");

	generateInterleaved(Ain, Asize);
	
}

void dpFFT::memoryCopyOut(){
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err); 
	clErrChk(err);

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
	//clErrChk(clFinish(queue));
	
}

int dpFFT::execute(){
	err=clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL);
	clErrChk(err);
	clErrChk(clFinish(queue));
	
	if (err <0)
		return -1;

	return 0;

}

void dpFFT::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Aout, 0, NULL, NULL));

	//clErrChk(clFinish(queue));
}

void dpFFT::cleanUp(){
	clErrChk(clReleaseMemObject(buffer));
	clErrChk(clfftDestroyPlan(&planHandle));
	free(Aout);
	free(Ain);
	clErrChk(clfftTeardown());
}

void dpFFT::generateInterleaved(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < 2*N; i=i+2){
		A[i] = rand() / (RAND_MAX/99999.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99999.9 + 1);
	}
}
*/
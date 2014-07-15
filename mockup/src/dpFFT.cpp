#include "dpFFT.hpp"
#include "helperFunctions/errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }


dpFFT::dpFFT(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
}

void dpFFT::init(int filler1, int filler2, int filler3){
	
	Asize = 16384;
	
	Ain = new float[Asize*2];
	Aout = new float[Asize*2];
	if (!Aout || !Ain)
		fprintf(stderr,"error in dynamic allocation");
	
	generateInterleaved(Ain, Asize);
	clErrChk(clfftSetup(&fftSetup));
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err); clErrChk(err);

}

void dpFFT::memoryCopyOut(){
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
	delete[] Ain;
	clErrChk(clfftDestroyPlan(&planHandle));
	clfftTeardown();
}

void dpFFT::generateInterleaved(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < 2*N; i=i+2){
		A[i] = rand() / (RAND_MAX/99999.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99999.9 + 1);
	}
}
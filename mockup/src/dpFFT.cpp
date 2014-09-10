#include "dpFFT.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }


dpFFT::dpFFT(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
	name = "FFT";
	localSize[0] =localSize[1]=localSize[2]=0;
	
	clErrChk(clfftSetup(&fftSetup));

}

void dpFFT::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] =localSize[1]=localSize[2]=1;
	for (int i = 0; pow(2,i) *2 *sizeof(float)/(float) 1048576 <= dataMB; i++)
		Asize = pow(2,i);
	
	MB = Asize*2*sizeof(float)/1048576;
}

void dpFFT::init(){
	dataParameters.push_back(Asize);
	dataNames.push_back("nVectors");
	
	Ain = new float[Asize*2];
	Aout = new float[Asize*2];
	if (!Aout || !Ain)
		fprintf(stderr,"error in dynamic allocation");
	
	generateInterleaved(Ain, Asize);
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

int dpFFT::execute(){
	err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL);
	clErrChk(err);
	if (err <0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
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
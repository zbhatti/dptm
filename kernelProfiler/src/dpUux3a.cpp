#include "dpUux3a.hpp"
#include "errorCheck.hpp"
#include "cmplx.h"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#include <string.h>

dpUux3a::dpUux3a(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	name = "Uux3a";
	
	std::string programSource = getFile("./src/cl/Uux3a_C.cl");
	kernelString = programSource.c_str();
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clErrChk(err);
	programCheck(err, context, program);
	
	kernel = clCreateKernel(program, "Uux3a", &err); clErrChk(err);
}

void dpUux3a::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	nEvents = 1048576*dataMB/(sizeof(float)*5*4);
	
	//need globalSize = nEvents to be divisible by localSize which is a power of 2:
	int n = 0;
	n = floor(log(nEvents) / log(2))
	nEvents = pow(2,n);
	
	MB = ( nEvents * sizeof(float)*5*4) / 1048576;

}

void dpUux3a::init(){
	//allocate local memory for original array
	eventsP = new float[5*4*nEvents]; //4 momentum for each of the 5 particles in an event. nEevents
	Amp = new cmplx[nEvents]; //complex "ampsum" for each event. nEvents
	inputBytes = 5*4*nEvents*sizeof(float);
	outputBytes = nEvents*sizeof(cmplx);
	
	if(!eventsP || !Amp)
		fprintf(stderr, "error in malloc\n");
	
	generateArray(eventsP, nEvents);

	dataParameters.push_back(nEvents);
	dataNames.push_back("nEvents");
}

void dpUux3a::memoryCopyOut(){
	eventsP_d = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBytes, NULL, &err); clErrChk(err);
	Amp_d = clCreateBuffer(context, CL_MEM_READ_WRITE, outputBytes, NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, eventsP_d, CL_TRUE, 0, inputBytes, eventsP, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
}

void dpUux3a::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &eventsP_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Amp_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &nEvents));
	
	globalSize[0] = nEvents;
	
	
}

int dpUux3a::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpUux3a::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Amp_d, CL_TRUE, 0, outputBytes, Amp, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
	
	
	for (int i=0; i<1; i++){ //events
		
		if ((Amp[i].re ==0.) && (Amp[i].im == 0.))
			continue;
		
		printf("event %d:\n", i+1);
		for (int j=0; j<5; j++){ //particle id
			printf("particle %d:\n", j+1);
			for (int k=0; k<4; k++){ //4 momentum
				printf("%d: %f, ", k, eventsP[i*5*4 + 4*j + k]);
			}
			printf("\n");
		}
		
		printf("amp: %f + %fi\n", Amp[i].re, Amp[i].im);
	}
	
	
}

void dpUux3a::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(eventsP_d));
	clErrChk(clReleaseMemObject(Amp_d));
	delete[] eventsP;
	delete[] Amp;
}

void dpUux3a::generateArray(float *eventsP, int nEvents){
	int n,j,k;
	srand(time(NULL));
	
	for (n=0; n < nEvents; n++){
		for (j=0; j<5; j++){
			for (k=0; k<4; k++){
				//eventsP[n*5*4 + 4*j + k]=rand() / (RAND_MAX/999.9 + 1);
				eventsP[n*5*4 + 4*j + k]= (j+1)*(k + 1);
			}
		}
	}
}
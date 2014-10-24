#include "dpOxxxxx.hpp"
#include "errorCheck.hpp"
#include "cmplx.h"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#include <string.h>

dpOxxxxx::dpOxxxxx(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	name = "Oxxxxx";
	
	std::string programSource = getFile("./src/cl/oxxxxx_C.cl");
	kernelString = programSource.c_str();
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clErrChk(err);
	programCheck(err, context, program);
	
	kernel = clCreateKernel(program, "Oxxxxx", &err); clErrChk(err);
}

void dpOxxxxx::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	Psize = 1048576*dataMB/(sizeof(double)*4);
	MB = Psize * (sizeof(double)*4) / 1048576;
}

void dpOxxxxx::init(){

	P = new double[4*Psize];
	Fo = new cmplx[6*Psize];
	
	if(!P || !Fo)
		fprintf(stderr, "error in malloc\n");
	
	generateArray(P, Psize);
	
	dataParameters.push_back(Psize);
	dataNames.push_back("nElements");
}

void dpOxxxxx::memoryCopyOut(){
	P_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Psize*sizeof(double)*4, NULL, &err); clErrChk(err);
	Fo_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Psize*sizeof(cmplx)*6, NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, P_d, CL_TRUE, 0, Psize*sizeof(double)*4, P, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
}

void dpOxxxxx::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &P_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Fo_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Psize));

	globalSize[0] = Psize;
}

int dpOxxxxx::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpOxxxxx::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Fo_d, CL_TRUE, 0, Psize*sizeof(cmplx)*6, Fo, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpOxxxxx::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(P_d));
	clErrChk(clReleaseMemObject(Fo_d));
	delete[] P;
	delete[] Fo;
}

void dpOxxxxx::generateArray(double *P, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N - 4; i=i+4){
		P[i+0]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+1]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+2]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+3]=rand() / (RAND_MAX/99999.9 + 1);
	}
}
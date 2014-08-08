#include "dpSquareArray.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpSquareArray::dpSquareArray(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;

	name = "SquareArray";
	kernelString = "\n"
		"__kernel void squareArray( \n" 
		"	__global const float * Ain_d, \n"
		"	__global float * Aout_d, \n"
		"	const int N){ \n"
		"		int idx = get_global_id(0); \n"
		"		if (idx<N){ \n"
		"			Aout_d[idx] = Ain_d[idx] * Ain_d[idx]; \n"
		"		} \n"
		"	} \n";

	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "squareArray", &err); clErrChk(err);
}

void dpSquareArray::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	for(int i=0; pow(2,i)*sizeof(float)/(float) 1048576<dataMB;i++)
		Asize = pow(2,i);
	
	MB = Asize * sizeof(float) / (float) 1048576;
	
}

void dpSquareArray::init(){
	Ain = new float[Asize];
	Aout = new float[Asize];
	if (!Aout || !Ain)
		fprintf(stderr, "error in dynamic allocation");
	
	generateArray(Ain, Asize);
	
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");
	
}

void dpSquareArray::memoryCopyOut(){
	Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpSquareArray::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Asize));
	globalSize[0] = Asize;
}

int dpSquareArray::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL); 
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpSquareArray::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpSquareArray::cleanUp(){
	//printf("%f^2 = %f\n",Ain[Asize-1],Aout[Asize-1]);
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(Ain_d));
	clErrChk(clReleaseMemObject(Aout_d));
	delete[] Aout;
	delete[] Ain;
}

void dpSquareArray::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}

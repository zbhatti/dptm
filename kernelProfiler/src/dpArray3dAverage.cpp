#include "dpArray3dAverage.hpp"
#include "errorCheck.hpp"
#include <numeric>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpArray3dAverage::dpArray3dAverage(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = THREE_D;
	name = "Array3dAverage";

	
	kernelString ="\n"
	"__kernel void average3d(__global const float * Ain_d, __global float * Aout_d, const int N){  \n"
	"	int idx = get_global_id(0);                                                                  \n"
	"	int idy = get_global_id(1);                                                                  \n"
	"	int idz = get_global_id(2);                                                                  \n"
	"	                                                                                             \n"
	"	int idg = idx + idy*N + idz*N*N;                                                             \n"
	"	                                                                                             \n"
	"	int idTop= idx + idy*N + (idz-1)*N*N;                                                        \n"
	"	int idBottom= idx + idy*N + (idz+1)*N*N;                                                     \n"
	"	int idLeft= (idx-1) + idy*N + idz*N*N;                                                       \n"
	"	int idRight= (idx+1) + idy*N + idz*N*N;                                                      \n"
	"	int idFront= idx + (idy-1)*N + (idz)*N*N;                                                    \n"
	"	int idBack= idx + (idy+1)*N + (idz)*N*N;                                                     \n"
	"	                                                                                             \n"
	"	int neighbors[6]={idTop, idBottom, idLeft, idRight, idFront, idBack};                        \n"
	"	                                                                                             \n"
	"	float Sum = Ain_d[idg];                                                                      \n"
	"	int numNeighbors = 0;                                                                        \n"
	"	for(int i = 0; i < 6; i++){                                                                  \n"
	"		if((neighbors[i] < 0)||(neighbors[i] >N*N*N))                                              \n"
	"			continue;                                                                                \n"
	"		Sum = Sum + Ain_d[neighbors[i]];                                                           \n"
	"		numNeighbors ++;                                                                           \n"
	"	}                                                                                            \n"
	"	                                                                                             \n"
	"	Aout_d[idg] = Sum/(float)numNeighbors;                                                       \n"
	"	                                                                                             \n"
	"}                                                                                             \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err =clBuildProgram(program, 0, NULL, NULL, NULL, NULL); clErrChk(err);
	programCheck(err, context, program);
	kernel = clCreateKernel(program, "average3d", &err); clErrChk(err);
	
}

void dpArray3dAverage::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	int MAX;
	unsigned int bytes = dataMB * 1048576;
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = zLocal;
	
	MAX = std::max(xLocal, std::max(yLocal, zLocal) );
	Alength = MAX;
	
	//scan up to MB size specified to set dimensions of 3d array
	if (Alength*Alength*Alength*sizeof(float) > bytes)
		Alength = 8;
	
	else{
		for (; pow(Alength + MAX,3)*sizeof(float)<= bytes;){
			Alength = Alength + MAX;
		}
	}
	
	MB = Alength*Alength*Alength*sizeof(float) / (float) 1048576;
}


void dpArray3dAverage::init(){
	nElements = Alength*Alength*Alength;
	Ain = new float[nElements];
	Aout = new float[nElements];
	if (!Aout || !Ain)
		fprintf(stderr, "error in dynamic allocation");
	
	generate3dArray(Ain, Alength);

	dataParameters.push_back(Alength);
	dataNames.push_back("cubeLength");
}
 
void dpArray3dAverage::memoryCopyOut(){
	Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nElements*sizeof(float), NULL, &err); clErrChk(err);
	Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nElements*sizeof(float), NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, nElements*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpArray3dAverage::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Alength));
	
	globalSize[0] = Alength;
	globalSize[1] = Alength;
	globalSize[2] = Alength;
}

int dpArray3dAverage::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize, localSize, 0, NULL, NULL); 
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpArray3dAverage::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, nElements*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpArray3dAverage::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(Ain_d));
	clErrChk(clReleaseMemObject(Aout_d));
	delete[] Aout;
	delete[] Ain;
}

void dpArray3dAverage::generate3dArray(float *A, int N){
	int idx,idy,idz;
	srand(time(NULL));
	for (idx=0; idx < N; idx++){
		for (idy=0; idy < N; idy++){
			for (idz=0; idz < N; idz++){
				A[idx + idy*N + idz*N*N] = rand() / (RAND_MAX/99999.9 + 1);
			}
		}
	}
}

// http://programmingknowledgeblog.blogspot.com/2013/04/c-program-to-find-hcf-n-lcm-of-two.html

int dpArray3dAverage::lcm(int a, int b){
	int c = a * b;
	while (a != b){
		if (a > b)
			a = a - b;
		else
			b = b - a;
	}
	return c/a;

}


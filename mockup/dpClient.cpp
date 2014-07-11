#include <stdio.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <vector>
#include <math.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include "dpFFT.hpp"
#include "dpSquareArray.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpRotateImage.hpp"
#include "dpTiming.hpp"

class dpClient {
	cl_platform_id platform_ids[16];
	cl_device_id device_ids[16];
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	char * kernelString;
	size_t MaxWorkGroupSize;
	int MaxComputeUnits;
	struct timeval start, finish;
	std::vector<dpKernel*> taskList;
	std::vector<dpTiming> timeList;
	
	public:
		dpClient(int,int);
		void generateArray(float[],int);
		void generateMatrix(float[],int,int);
		void generateInterleaved(float[],int);
		void printMatrix(float[],int,int);
		void printInterlaved(float[], int);
		float timeDiff(struct timeval, struct timeval);
		int MaxWGSize(){return MaxWorkGroupSize;};
		void runSquareArray();
		void runFFT();
		void runMatrixMultiplication();
		void runRotateImage();
		void runKernels();
		void printTimes();
		
};

//set up context and queue on a device and retrieve valuable
//device information for other methods
dpClient::dpClient(int platform, int device){
	unsigned int numDevices;
	char name[256];
	int err;
	cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,0,0};
	err = clGetPlatformIDs(16, platform_ids, NULL);
	err = clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, 16, device_ids, &numDevices);
	err = clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"On Platform %s\n", name);
	err = clGetDeviceInfo(device_ids[device], CL_DEVICE_NAME, sizeof(name), name, NULL);
	err = clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, NULL);
	err = clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MaxComputeUnits), &MaxComputeUnits, NULL);
	fprintf(stderr,"using device %s\n", name);
	props[1] = (cl_context_properties) platform_ids[platform];
	context = clCreateContext(props, 1, &device_ids[device], NULL, NULL, &err);
	queue = clCreateCommandQueue( context, device_ids[device], 0, &err);
}

void dpClient::runSquareArray(){
	int Asize = 1024;
	float* Ain = new float[Asize];
	generateArray(Ain, Asize);
	dpSquareArray squareKernel(Ain, Asize, context, queue, MaxWorkGroupSize);
	squareKernel.memoryCopyOut();
	squareKernel.plan();
	squareKernel.execute();
	squareKernel.memoryCopyIn();
	squareKernel.cleanUp();
	delete[] Ain;
}

void dpClient::runFFT(){
	int Asize =1024;
	float* Ain = new float[Asize*2];
	generateInterleaved(Ain, Asize);
	dpFFT FFTKernel(Ain, Asize, context, queue);
	FFTKernel.memoryCopyOut();
	FFTKernel.plan();
	FFTKernel.execute();
	FFTKernel.memoryCopyIn();
	FFTKernel.cleanUp();
	delete[] Ain;
}

void dpClient::runMatrixMultiplication(){
	int N, P, M;
	float *A, *B;
	N=1024;
	P=1024;
	M=1024;
	A = new float[N*P];
	B = new float[P*M];
	generateMatrix(A,N,P);
	generateMatrix(B,P,M);
	dpMatrixMultiplication MMKernel(A, B, N, P, M, context, queue, 32, 16);
	MMKernel.memoryCopyOut();
	MMKernel.plan();
	MMKernel.execute();
	MMKernel.memoryCopyIn();
	MMKernel.cleanUp();
	delete[] A;
	delete[] B;
	
}

void dpClient::runRotateImage(){
	dpRotateImage rotKernel(context, queue, 16, 16);
	rotKernel.memoryCopyOut();
	rotKernel.plan();
	rotKernel.execute();
	rotKernel.memoryCopyIn();
	rotKernel.cleanUp();
}

void dpClient::runKernels(){

	int N1, N2, N3; 
	int P1, P2, P3; 
	int M1, M2, M3;
	float *A1, *A2, *A3; 
	float *B1, *B2, *B3;
	
	N1=512; N2=1024; N3=256;
	P1=4096; P2=2048; P3=4096;
	M1=2048; M2=8192; M3=8192;
	
	A1 = new float[N1*P1]; A2 = new float[N2*P2]; A3 = new float[N3*P3];
	B1 = new float[P1*M1]; B2 = new float[P2*M2]; B3 = new float[P3*M3];
	
	generateMatrix(A1,N1,P1); generateMatrix(A2,N2,P2); generateMatrix(A3,N3,P3);
	generateMatrix(B1,P1,M1); generateMatrix(B2,P2,M2); generateMatrix(B3,P3,M3);
	
	dpMatrixMultiplication MM1(A1, B1, N1, P1, M1, context, queue, 4, 16);
	dpMatrixMultiplication MM2(A2, B2, N2, P2, M2, context, queue, 16, 16);
	dpMatrixMultiplication MM3(A3, B3, N3, P3, M3, context, queue, 8, 32);

	
	int C1size, C2size, C3size;
	C1size=1024; C2size=8192; C3size=1048576;
	float *C1, *C2, *C3;
	C1 = new float[C1size]; C2 = new float[C2size]; C3 = new float[C3size];
	generateArray(C1, C1size); generateArray(C2, C2size); generateArray(C3, C3size);
	dpSquareArray square1(C1, C1size, context, queue, 256); //C1size must be larger than xLocal
	dpSquareArray square2(C2, C2size, context, queue, 16);
	dpSquareArray square3(C3, C3size, context, queue, MaxWorkGroupSize);
	
	dpRotateImage rot1(context, queue, 16, 16);
	dpRotateImage rot2(context, queue, 8, 32);
	dpRotateImage rot3(context, queue, 64, 4);
	
	taskList.push_back(&MM1);
	taskList.push_back(&MM2);
	taskList.push_back(&MM3);
	taskList.push_back(&square1);
	taskList.push_back(&square2);
	taskList.push_back(&square3);
	taskList.push_back(&rot1);
	taskList.push_back(&rot2);
	taskList.push_back(&rot3);
	
	dpTiming timeTmp;
	
	for (int i =0; i <taskList.size(); i++){
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyOut();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyOut = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->plan();
		gettimeofday(&finish, NULL);
		timeTmp.plan = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->execute();
		gettimeofday(&finish, NULL);
		timeTmp.execute = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyIn();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyIn = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->cleanUp();
		gettimeofday(&finish, NULL);
		timeTmp.cleanUp = timeDiff(start,finish);
		
		timeList.push_back(timeTmp);
		
	}
	
	delete[] C1, C2, C3;
	delete[] A1, A2, A3;
	delete[] B1, B2, B3;
}

void dpClient::generateArray(float A[], int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}

void dpClient::generateMatrix(float A[], int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}

void dpClient::generateInterleaved(float A[], int N){
	int i;
	srand(time(NULL));
	for (i=0; i < 2*N; i=i+2){
		A[i] = rand() / (RAND_MAX/99999.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99999.9 + 1);
	}
}

float dpClient::timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000.0 + finish.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec))/(1000.0);
}

//helper function:
void dpClient::printMatrix(float A[], int height, int width){
	int i, j;
	printf("\n");
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			printf("%3.2f ",A[i + width*j]);
		}
		printf("\n");
	}
	printf("\n");
}

//helper function:
void dpClient::printInterlaved(float A[], int N){
	int i;
	for (i=0; i < 2*N; i=i+2)
		printf("%f, %f\n",A[i], A[i+1]);
}

//print times, probably change to export the timeList instance
void dpClient::printTimes(){
	for (int i = 0; i < timeList.size(); i++){
		printf("%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n", 
			timeList.at(i).memoryCopyOut,
			timeList.at(i).plan,
			timeList.at(i).execute,
			timeList.at(i).memoryCopyIn,
			timeList.at(i).cleanUp);
	}
}

int main(){
	dpClient cli1(0,0);
	cli1.runKernels();
	cli1.printTimes();
	
	dpClient cli2(1,0);
	cli2.runKernels();
	cli2.printTimes();
	
	dpClient cli3(1,1);
	cli3.runKernels();
	cli3.printTimes();
	
	dpClient cli4(2,0);
	cli4.runKernels();
	cli4.printTimes();
	
	dpClient cli5(2,1);
	cli5.runKernels();
	cli5.printTimes();
	
	/*
	int j=0;
	for(r=0; r<15; r++){
		for(i=0; pow(2,i)*pow(2,j)<=1024;i++){
			for(j=0; pow(2,i)*pow(2,j)<=1024;j++){
				cli1.matrixMultiplication(2048,2048,2048,pow(2,i),pow(2,j));
			}
			j=0;
		}
	}*/
	
	/*
	cli1.FFT(8);
	cli1.FFT(8192);
	for(r=0; r<2; r++){
		for(i=0; pow(2,i)*pow(2,j)<=cli1.MaxWGSize();i++){
			for(j=0; pow(2,i)*pow(2,j)<=cli1.MaxWGSize();j++){
				cli1.rotateImage(pow(2,i),pow(2,j));
			}
			j=0;
		}
	}
	
	for(r=0; r<2; r++){
		for(i=0; pow(2,i)*pow(2,j)<=cli2.MaxWGSize();i++){
			for(j=0; pow(2,i)*pow(2,j)<=cli2.MaxWGSize();j++){
				cli2.rotateImage(pow(2,i),pow(2,j));
			}
			j=0;
		}
	}
	*/
	return 0;
}








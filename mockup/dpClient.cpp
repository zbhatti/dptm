#include <stdio.h>
#include <stdlib.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include <new>
#include <math.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include <clFFT.h>
#include "helperFunctions/errorCheck.hpp"
#include "helperFunctions/bmpfuncs.h"

#include "dpFFT.hpp"
#include "dpSquareArray.hpp"
#include "dpMatrixMultiplication.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }


//using namespace std;

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
	//std::vector<clKernel> taskList needs null constructor and '='
	//use vector of <*dpKernel> instead
	//1D scan, 2D scan etc
	//timing in kernels
	//
	
	public:
		dpClient(int,int);
		void rotateImage(int, int);
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
		gettimeofday(&start, NULL);
	MMKernel.execute();
		gettimeofday(&finish, NULL);
		prinft("%f",timeDiff(start,finish))
	MMKernel.memoryCopyIn();
	MMKernel.cleanUp();
	delete[] A;
	delete[] B;
	
}


void dpClient::rotateImage(int threadsX, int threadsY){

	// Set the image rotation (in degrees)
	float theta = 3.14159/6;
	float cos_theta = cosf(theta);
	float sin_theta = sinf(theta);
	int imageHeight;
	int imageWidth;
	const char* inputFile = "helperFunctions/input.bmp";
	const char* outputFile = "helperFunctions/output.bmp";
	float* inputImage = readImage(inputFile, &imageWidth, &imageHeight);

	// Size of the input and output images on the host
	int dataSize = imageHeight*imageWidth*sizeof(float);

	// Output image on the host
	float* outputImage = new float[imageHeight*imageWidth];

	// Create the input and output buffers
	cl_mem d_input;
	cl_mem d_output;
	cl_int err;
	d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &err);  clErrChk(err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL,	&err);  clErrChk(err);

	// Copy the input image to the device
	clErrChk(clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, dataSize, inputImage, 0, NULL, NULL));

	// Create the kernel object
	kernel = clCreateKernel(program, "img_rotate", &err); clErrChk(err);
	
	// Set the kernel arguments
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_input));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &imageWidth));
	clErrChk(clSetKernelArg(kernel, 3, sizeof(int), &imageHeight));
	clErrChk(clSetKernelArg(kernel, 4, sizeof(float), &sin_theta));
	clErrChk(clSetKernelArg(kernel, 5, sizeof(float), &cos_theta));

	// Set the work item dimensions
	size_t globalSize[2] = {imageWidth, imageHeight};
	size_t localSize[2] = {threadsX, threadsY};
	gettimeofday(&start, NULL);	
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	gettimeofday(&finish, NULL);
	//printf("height,localSizeX,localSizeY,execution\n");
	printf("%d,%d,%d,%3.3f\n",imageHeight,localSize[0],localSize[1],timeDiff(start,finish));

	// Read the image back to the host
	clErrChk(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dataSize, outputImage, 0, NULL, NULL));

	// Write the output image to file
	//storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseMemObject(d_input));
	clErrChk(clReleaseMemObject(d_output));
	delete[] outputImage;
}

//void dpClient::convolve(int type, int xThreads, int yThreads){
	//opencl variables
	//build filter
	
	//copy memory to device
	
	//set kernel arguments
	
	//launch kernel
	
	//copy memory back to host
	
	//switch on type: 1-set,2-arithmetic,3-globalconv,4-localconv}

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

int main(){
	dpClient cli3(0,0);
	cli3.runSquareArray();
	cli3.runFFT();
	cli3.runMatrixMultiplication();
	
	int j=0;
	/*
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








#include <stdio.h>
#include <stdlib.h>
#include <time.h> //for random seed and timing
#include <sys/time.h>
#include "helperFunctions/errorCheck.hpp"
#include <new>
#include <math.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include <clFFT.h>
#include "helperFunctions/bmpfuncs.h"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

using namespace std;

class clClient {
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
		clClient(int,int);
		void loadProgram();
		void squareArray(int);
		void matrixMultiplication(int,int,int,int,int);
		void FFT(int);
		void rotateImage(int, int);
		void generateArray(float[],int);
		void generateMatrix(float[],int,int);
		void generateInterleaved(float[],int);
		void printMatrix(float[],int,int);
		void printInterlaved(float[], int);
		float timeDiff(struct timeval, struct timeval);
		int MaxWGSize(){return MaxWorkGroupSize;};
		
};

//set up context and queue on a device and retrieve valuable
//device information for other methods
clClient::clClient(int platform, int device){
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

//load the text file containing kernels and build the program 
//which allows other methods to launch kernels found in the text file
void clClient::loadProgram(){
	int err;
	long fileLength;
	const char fileName[] = "./kernels.cl";
	FILE *fp = fopen(fileName, "r");
	if(!fp){
		fprintf(stderr, "error in opening file!");
	}else{}
  fseek (fp, 0, SEEK_END);
  fileLength = ftell(fp);
  rewind (fp);
	kernelString = new char[fileLength];
	fread (kernelString, 1, fileLength, fp);
	fclose(fp);
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
}

//squares the elements in an array
void clClient::squareArray(int Asize){
	int err, i;
	float* Ain, *Aout; 
	cl_mem Ain_d, Aout_d;
	size_t localSize, globalSize;
	
	//preprocessing
	kernel = clCreateKernel(program, "squareElements", &err);
	Ain = new float[Asize];
	Aout = new float[Asize];
	if(!Ain || !Aout) 
		fprintf(stderr,"error in dynamic allocation");
	generateArray(Ain, Asize);
	
	//memory copy out
	Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
	//planning
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Asize));
	localSize = (size_t) MaxWorkGroupSize;
	globalSize = Asize - Asize%localSize + localSize;;
	
	//execution
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
	//memory copy in
	clErrChk(clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
	//clearing memory
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseMemObject(Ain_d));
	clErrChk(clReleaseMemObject(Aout_d));
	delete[] Ain;
	delete[] Aout;
	
}

//does the muliplication A[N][P]*B[P][M] = C[N][M]
void clClient::matrixMultiplication(int N, int P, int M, int threadsX, int threadsY){
	///Declare and Initialize Data
	float *A, *B, *C; 
	int err, szA, szB, szC;
	size_t global[2]; 
	size_t local[2]; 
	cl_mem a_in, b_in, c_out; 
	
	//preprocesing
	szA = N*P; 
	szB = P*M; 
	szC = N*M; 
	A = new float[szA]; 
	B = new float[szB]; 
	C = new float[szC];
	if (!A || !B || !C)
		fprintf(stderr,"error in dynamic allocation");
	generateMatrix(A,N,P);
	generateMatrix(B,P,M);
	
	//memory copy out
	a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szA, NULL, &err); clErrChk(err);
	b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szB, NULL, &err); clErrChk(err);
	c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC, NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(float) * szA, A, 0, NULL, NULL)); 
	clErrChk(clEnqueueWriteBuffer(queue, b_in, CL_TRUE, 0, sizeof(float) * szB, B, 0, NULL, NULL));
	clErrChk(clFinish(queue));

	//planning
	kernel = clCreateKernel(program, "mmul", &err); clErrChk(err);
	clErrChk(clSetKernelArg(kernel, 0, sizeof(int), &M)); 
	clErrChk(clSetKernelArg(kernel, 1, sizeof(int), &N)); 
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &P)); 
	clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_in)); 
	clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_in)); 
	clErrChk(clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_out));
	global[0] = (size_t) M; global[1] = (size_t) N;
	local[0] = (size_t) threadsX; local[1] = (size_t) threadsY; 
	
	//Execution
	gettimeofday(&start, NULL);	
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	gettimeofday(&finish, NULL);
	//printf("Predicted local[0]: %d, local[1]: %d\n", ,);
	
	//printf("height,width,localSizeX,localSizeY,execution\n");
	printf("%d,%d,%d,%3.3f\n",N,local[0],local[1],timeDiff(start,finish));
	
	//memory copy in
	clErrChk(clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(float) * szC, C, 0, NULL, NULL ));
	clErrChk(clFinish(queue));
	
	//release memory
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseMemObject(a_in));
	clErrChk(clReleaseMemObject(b_in));
	clErrChk(clReleaseMemObject(c_out));
	delete[] A;
	delete[] B;
	delete[] C;
}

//uses clFFT library
void clClient::FFT(int Asize){
	float *Ain, *Aout;
	
	//declaring general opencl variables
	cl_mem buffer;
	cl_int err;
	
	//clFFT variables
	clfftPlanHandle planHandle;
	clfftSetupData fftSetup;
	err = clfftSetup(&fftSetup);
	size_t clLengths[1];
	
	//Preprocessing
	Ain = new float[Asize*2];
	Aout = new float[Asize*2];
	if (!Aout || !Ain)
		fprintf(stderr,"error in dynamic allocation");
	generateInterleaved(Ain, Asize);

	//memcpyOut
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));

	//plan
	clLengths[0] = (size_t) Asize;
	clErrChk(clfftCreateDefaultPlan(&planHandle, context, CLFFT_1D, clLengths));
	clErrChk(clfftSetPlanPrecision(planHandle, CLFFT_SINGLE));
	clErrChk(clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
	clErrChk(clfftSetResultLocation(planHandle, CLFFT_INPLACE));
	clErrChk(clfftBakePlan(planHandle, 1, &queue, NULL, NULL));
	clErrChk(clFinish(queue));

	//execute
	clErrChk(clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL));
	clErrChk(clFinish(queue));

	//memcpyIn
	clErrChk(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
	
	//release memory
	clErrChk(clReleaseMemObject(buffer));
	delete[] Ain;
	delete[] Aout;
	clErrChk(clfftDestroyPlan(&planHandle));
	clfftTeardown();
}

void clClient::rotateImage(int threadsX, int threadsY){

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

void clClient::convolve(int type, int xThreads, int yThreads){
	//opencl variables
	//build filter
	
	//copy memory to device
	
	//set kernel arguments
	
	//launch kernel
	
	//copy memory back to host
	
	//switch on type: 1-set,2-arithmetic,3-globalconv,4-localconv
	
}

void clClient::generateArray(float A[], int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}

void clClient::generateMatrix(float A[], int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}

void clClient::generateInterleaved(float A[], int N){
	int i;
	srand(time(NULL));
	for (i=0; i < 2*N; i=i+2){
		A[i] = rand() / (RAND_MAX/99999.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99999.9 + 1);
	}
}

float clClient::timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000.0 + finish.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec))/(1000.0);
}

//helper function:
void clClient::printMatrix(float A[], int height, int width){
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
void clClient::printInterlaved(float A[], int N){
	int i;
	for (i=0; i < 2*N; i=i+2)
		printf("%f, %f\n",A[i], A[i+1]);
}


int main(){
	clClient cli1(0,0);
	clClient cli2(1,0);
	cli1.loadProgram();
	cli1.squareArray(128);
	cli1.squareArray(16777216);
	
	
	int i,j,r;
	j =0;
	/*
	for(r=0; r<15; r++){
		for(i=0; pow(2,i)*pow(2,j)<=1024;i++){
			for(j=0; pow(2,i)*pow(2,j)<=1024;j++){
				cli1.matrixMultiplication(2048,2048,2048,pow(2,i),pow(2,j));
			}
			j=0;
		}
	}*/
		
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
	
		
	return 0;
}








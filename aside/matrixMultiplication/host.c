//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
#include <sys/time.h>
#include <stdio.h>
#include "../../utilities/src/errorCheck.h"
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); timeSpent=timeDiff(start, finish); printf("%0.3f,",timeSpent);

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

void initMat(int Mdim, int Ndim, int Pdim, float* A, float* B, float* C){
	int i, j, k;
	//Generate A:
	for (i=0; i< Ndim; i++){//rows in A
		for (j=0; j<Pdim; j++){//cols in A
			A[i + Ndim*j] = rand()% 100000;
		}
	}
	
	//Generate B:
	for (i=0; i< Pdim; i++){//rows in B
		for (j=0; j<Mdim; j++){//cols in B
			B[i + Pdim*j] = rand()% 100000;
		}
	}

	//Clear C:
	for (i=0; i< Ndim; i++){//rows in C
		for (j=0; j<Mdim; j++){//cols in C
			C[i + Ndim*j] = 0.0;
		}
	}
}

void printMatrix(float *A, int height, int width){
	int i, j;
	printf("\n");
	for (i=0; i<height; i++){//rows in A
		for (j=0; j<width; j++){//cols in A
			printf("%1.1f ",A[i + height*j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char **argv) {
	///Declare and Initialize Data
	float *A, *B, *C; 
	int Mdim, Ndim, Pdim; 
	int err, szA, szB, szC;
	float timeSpent;
	struct timeval start, finish;
	char name[256];
	size_t global[2]; 
	size_t local[2]; 
	cl_device_id device_ids[16];
	cl_platform_id platform_ids[16];
	unsigned int numDevices;
	cl_context context;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0};
	cl_command_queue commands; 
	cl_program program; 
	cl_kernel kernel; 
	cl_uint nd; 
	cl_mem a_in, b_in, c_out; 
	Ndim = 9; //leads to each matrix being ~67MB 4096
	Pdim = 8; 
	Mdim = 10; 
	szA = Ndim*Pdim; 
	szB = Pdim*Mdim; 
	szC = Ndim*Mdim; 
	A = (float *)malloc(szA*sizeof(float)); 
	B = (float *)malloc(szB*sizeof(float)); 
	C = (float *)malloc(szC*sizeof(float));
	srand(time(NULL));
	initMat(Mdim, Ndim, Pdim, A, B, C); 
	printMatrix(A, Ndim,Pdim);
	printMatrix(B, Pdim, Mdim);

	
	int PlatformChoice=0;
	int DeviceChoice=0;
	int Fast=0;
	
	if (argc>1)
		PlatformChoice=atoi(argv[1]);
	
	if (argc>2)
		DeviceChoice=atoi(argv[2]);
	
	if (argc>3)
		Fast=atoi(argv[3]);
	
	
	//read kernel source file into a string
	char * kernelString;
	kernelString = (char*)malloc(0x100000);
	long fileLength;
	const char fileName[] = "./kernel.cl";
	FILE *fp = fopen(fileName, "r");
	if(!fp){
		fprintf(stderr, "error in opening file!");
	}else{}
  fseek (fp, 0, SEEK_END);
  fileLength = ftell(fp);
  rewind (fp);
	fread (kernelString, 1, fileLength, fp);
	fclose(fp);
	
	//getting information names of Platforms and Devices:
	clErrChk(clGetPlatformIDs(16, platform_ids, NULL));
	clErrChk(clGetDeviceIDs(platform_ids[PlatformChoice], CL_DEVICE_TYPE_ALL, 16, device_ids, &numDevices));
	clErrChk(clGetPlatformInfo(platform_ids[PlatformChoice], CL_PLATFORM_NAME, sizeof(name), name, NULL));
	fprintf(stderr,"On Platform %s\n", name);
	clErrChk(clGetDeviceInfo(device_ids[DeviceChoice], CL_DEVICE_NAME, sizeof(name), name, NULL));
	fprintf(stderr,"using device %s\n", name);
	props[1] = (cl_context_properties) platform_ids[PlatformChoice];
	context = clCreateContext(props, 1, &device_ids[DeviceChoice], NULL, NULL, &err); clErrChk(err);
	commands = clCreateCommandQueue( context, device_ids[DeviceChoice], 0, &err); clErrChk(err);
	
	BEGIN
	//setup bufferes and write A and B matrices to device memory
	a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szA, NULL, &err); clErrChk(err);
	b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szB, NULL, &err); clErrChk(err);
	c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC, NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(commands, a_in, CL_TRUE, 0, sizeof(float) * szA, A, 0, NULL, NULL)); 
	clErrChk(clEnqueueWriteBuffer(commands, b_in, CL_TRUE, 0, sizeof(float) * szB, B, 0, NULL, NULL));
	clErrChk(clFinish(commands));
	END
	
	BEGIN
	//Build the program, define the kernel and setup arguments
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	if(Fast){
		kernel = clCreateKernel(program, "mmulFast", &err); clErrChk(err);
	}
	else{
		kernel = clCreateKernel(program, "mmul", &err); clErrChk(err);
	}
	clErrChk(clSetKernelArg(kernel, 0, sizeof(int), &Mdim)); 
	clErrChk(clSetKernelArg(kernel, 1, sizeof(int), &Ndim)); 
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Pdim)); 
	clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_in)); 
	clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_in)); 
	clErrChk(clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_out));
	if(Fast)//set this argument depending on which kernel we are running
		clErrChk(clSetKernelArg(kernel, 6, sizeof(float)*Pdim, NULL));
	clErrChk(clFinish(commands));
	END
	
	BEGIN
	//Run the kernel and collect results
	if(Fast){
		global[0] = (size_t) Ndim; local[0] = (size_t) 256; //local[0] = Ndim/max compute units
		clErrChk(clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, local, 0, NULL, NULL));
	}
	else{
		global[0] = (size_t) Ndim; global[1] = (size_t) Mdim;
		clErrChk(clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, NULL));
	}
	clErrChk(clFinish(commands));
	END
	
	BEGIN
	clErrChk(clEnqueueReadBuffer(commands, c_out, CL_TRUE, 0, sizeof(float) * szC, C, 0, NULL, NULL ));
	clErrChk(clFinish(commands));
	END
	printf("\n");
	
	printMatrix(C, Ndim, Mdim);
	//test_results(A, B, c_out); 
} 
#include <clFFT.h>
#include <stdio.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#define ERRORLINE fprintf(stderr,"Line: %d\n",__LINE__);

const char* get_error_string(cl_int);
void clAssert(cl_int, const char*, int);



void generateInterleaved(float *A, int N){
	int i;
	for (i=0; i < 2*N; i=i+2){
		A[i] = rand() / (RAND_MAX/99999.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99999.9 + 1);
	}
}

int main(int argc, char *argv[]){
	cl_int err;
	cl_command_queue queue;
	cl_context context;
	cl_device_id device_ids[5];
	cl_platform_id platform_ids[5];
	clfftPlanHandle planHandle;
	clfftSetupData fftSetup;
	
	int DeviceChoice = 0;
	int PlatformChoice = 0;
	unsigned int num_devices;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	clErrChk(clGetPlatformIDs(5, platform_ids, NULL));
	clErrChk(clGetDeviceIDs(platform_ids[PlatformChoice], CL_DEVICE_TYPE_ALL, 5, device_ids, &num_devices));
	
	//starting up opencl on the selected device and platform:
	props[1] = (cl_context_properties) platform_ids[PlatformChoice];
	context = clCreateContext(props, 1, &device_ids[DeviceChoice], NULL, NULL, &err);
	clErrChk(err);
	queue = clCreateCommandQueue( context, device_ids[DeviceChoice], 0, &err);
	clErrChk(err);
	
	int repeat;
	for(repeat = 0; repeat < 40; repeat++, printf("%d",repeat)){
		float *Ain, *Aout;
		int Asize;
		cl_mem buffer;
		size_t clLengths[1];
		
		clErrChk(clfftSetup(&fftSetup));
		Asize = 500;
		Ain = (float*) malloc( sizeof(float)*Asize*2);
		Aout = (float*) malloc( sizeof(float)*Asize*2);
		if (!Aout || !Ain)
			fprintf(stderr,"error in dynamic allocation");
		
		generateInterleaved(Ain, Asize);

		buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err);
		clErrChk(err);
		
		
		clErrChk(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Ain, 0, NULL, NULL));
		clErrChk(clFinish(queue));

		clLengths[0] = (size_t) Asize;
		clErrChk(clfftCreateDefaultPlan(&planHandle, context, CLFFT_1D, clLengths));
		clErrChk(clfftSetPlanPrecision(planHandle, CLFFT_SINGLE));
		clErrChk(clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
		clErrChk(clfftSetResultLocation(planHandle, CLFFT_INPLACE));
		clErrChk(clfftBakePlan(planHandle, 1, &queue, NULL, NULL));
		clErrChk( clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL));
		clErrChk(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Aout, 0, NULL, NULL));
		clErrChk(clReleaseMemObject(buffer));
		clErrChk(clfftDestroyPlan(&planHandle));
		free(Aout);
		free(Ain);
		clfftTeardown();
	}

}



const char* get_error_string(cl_int err){
	switch(err){
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		default: return "Unknown OpenCL error";
	}
}

//code from stackexchange to print opencl return messages
void clAssert(cl_int code, const char *file, int line){
   if (code != 0){
      fprintf(stderr,"clErrChk: %s %s %d\n", get_error_string(code), file, line);
      //exit(code);
   }
}

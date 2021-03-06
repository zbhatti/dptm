#include "dpCudaNoMemory.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// Kernel that executes on the CUDA device
__global__ void NoMemory(){
	int b = 1;
}

//notice unused parameters for CUDA kernel:
dpCudaNoMemory::dpCudaNoMemory(cl_context ctx, cl_command_queue q){
	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "NoMemory";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaNoMemory::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = localSize[1] = localSize[2] = 1;
	MB = dataMB;
}

void dpCudaNoMemory::init(){}

void dpCudaNoMemory::memoryCopyOut(){}

void dpCudaNoMemory::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	nBlocks = 1024;
	END
}

int dpCudaNoMemory::execute(){
	cudaError_t err;
	BEGIN
	NoMemory <<< nBlocks, blockSize >>> ();
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	//printf("%0.3f,",delTime);
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaNoMemory::memoryCopyIn(){}

void dpCudaNoMemory::cleanUp(){}

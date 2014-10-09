#include "dpCudaComputation.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// Kernel that executes on the CUDA device
__global__ void Computation(float a, float b){
	float c = a*b;
}

//notice unused parameters for CUDA kernel:
dpCudaComputation::dpCudaComputation(cl_context ctx, cl_command_queue q){
	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "Computation";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaComputation::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = localSize[1] = localSize[2] = 1;

	MB = dataMB;
}

void dpCudaComputation::init(){
	// Allocate and initialize host arrays 
	srand(time(NULL));
	a = rand() / (RAND_MAX/1000);
	b = rand() / (RAND_MAX/1000);
	
	dataParameters.push_back(a);
	dataParameters.push_back(b);
	dataNames.push_back("a");
	dataNames.push_back("b");
}

void dpCudaComputation::memoryCopyOut(){
	BEGIN
	END
}

void dpCudaComputation::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	nBlocks = 1024;
	END
}

int dpCudaComputation::execute(){
	cudaError_t err;
	BEGIN
	Computation <<< nBlocks, blockSize >>> (a, b);
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaComputation::memoryCopyIn(){
	BEGIN
	END
}

void dpCudaComputation::cleanUp(){
}

void dpCudaComputation::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}
//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
__kernel void mmul( 
const int Mdim, const int Ndim, const int Pdim, 
__global float* A, __global float* B,  __global float* C){ 
	int k; 
	int i = get_global_id(0); 
	int j = get_global_id(1); 
	float tmp = 0.0f; 
	for (k=0; k<Pdim; k++) 
		tmp += A[i*Ndim+k] * B[k*Pdim+j]; 
	C[i*Ndim+j] += tmp; 
}

//
__kernel void mmulFast( 
const int Mdim, 
const int Ndim, 
const int Pdim, 
__global float* A, 
__global float* B, 
__global float* C, 
__local float* Bwrk){ //size of Bwrk is limited by device, but should be Pdim
	int k,j; 
	int i = get_global_id(0); 
	int iloc = get_local_id(0); 
	int nloc = get_local_size(0); 
	float Awrk[4096]; //k20x device has 65536 registers per block was Awrk[1000]
	float tmp;
	for (k=0; k<Pdim; k++) 
		Awrk[k] = A[i*Ndim+k]; 
	for (j=0; j<Mdim; j++){ 
		for (k=iloc; k<Pdim; k=k+nloc) 
			Bwrk[k] = B[k*Pdim+j]; 
		barrier(CLK_LOCAL_MEM_FENCE); 
		tmp = 0.0f; 
		for (k=0; k<Pdim; k++) 
			tmp += Awrk[k] * Bwrk[k]; 
		C[i*Ndim+j] += tmp; 
	}
}
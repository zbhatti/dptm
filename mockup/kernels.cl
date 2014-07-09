//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
__kernel void mmul( 
const int Mdim, 
const int Ndim,  
const int Pdim, 
__global float* A, 
__global float* B,  
__global float* C){ 
	int k; 
	int i = get_global_id(0); //get column
	int j = get_global_id(1); //get row
	float tmp = 0.0f; 
	for (k=0; k<Pdim; k++) 
		tmp += A[j*Pdim+k] * B[k*Mdim+i]; 
	C[j*Mdim+i] = tmp; 
}

//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
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

__kernel void squareElements(__global const float * Ain_d, __global float * Aout_d, const int N){
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	int idz = get_global_id(2);
	int Index = idx + idy*get_global_size(0) + idz*get_global_size(1)*get_global_size(2);
	//2D:int Index = idx + idy*get_global_size(0);
	
	if (Index<N){
		Aout_d[Index] = Ain_d[Index] * Ain_d[Index];
	}
}

//source: http://www.heterogeneouscompute.org/?page_id=7
__kernel 
void img_rotate(__global float* dest_data, 
                __global float* src_data,    
                           int  W,    
                           int  H, 
                         float  sinTheta, 
                         float  cosTheta) { 

   //Work-item gets its index within index space
   const int ix = get_global_id(0); 
   const int iy = get_global_id(1);    

   //Calculate location of data to move into (ix,iy) 
   //Output decomposition as mentioned
   float x0 = W/2.0f;
   float y0 = H/2.0f;

   float xOff = ix - x0;
   float yOff = iy - y0; 

   int xpos = (int)(xOff*cosTheta + yOff*sinTheta + x0 );
   int ypos = (int)(yOff*cosTheta - xOff*sinTheta + y0 ); 

   // Bounds Checking 
   if((xpos>=0) && (xpos< W) && (ypos>=0) && (ypos< H)) {

      // Read (ix,iy) src_data and store at (xpos,ypos) in 
      // dest_data
      // In this case, because we rotating about the origin
      // and there is no translation, we know that (xpos,ypos)  
      // will be unique for each input (ix,iy) and so each 
      // work-item can write its results independently
			
			//try taking away the global memory access done here. use local or private memory
      dest_data[iy*W+ix] = src_data[ypos*W+xpos];    
   }
}

//source: http://www.cs.uic.edu/~amusa/gpu/gpu_final.html
__kernel void convolve_naive(__global  float * pInput,
                       __global float * pFilter,
                       __global  float * pOutput,
                       int nInWidth,
                       int nFilterWidth)
{
    int nWidth = get_global_size(0);

    //printf((const char*)"%d ", nWidth);
    
    int xOut = get_global_id(0);
    int yOut = get_global_id(1);
    
    int xInTopLeft = xOut;
    int yInTopLeft = yOut;
    
    float sum = 0;
    for (int r = 0; r < nFilterWidth; r++)
    {
        int idxFtmp = r * nFilterWidth;
        
        int yIn = yInTopLeft + r;
        int idxIntmp = yIn * nInWidth + xInTopLeft;
        
        for (int c = 0; c < nFilterWidth; c++)
        {
            int idxF  = idxFtmp  + c;
            int idxIn = idxIntmp + c;
            sum += pFilter[idxF]*pInput[idxIn];
        }
    }
    int idxOut = yOut * nWidth + xOut;
    pOutput[idxOut] = sum;
}


//source: http://www.cs.uic.edu/~amusa/gpu/gpu_final.html
__kernel void convolve_local(const __global  float * pInput,
                       __constant float * pFilter,
                       __global  float * pOutput,
                       const int nInWidth,
                       const int nFilterWidth,
                       __local float * localImage)
{

    const int nWidth = get_global_size(0);

	int i = get_group_id(0);
	int j = get_group_id(1); 
	int idX = get_local_id(0);
	int idY = get_local_id(1);
	int ii = i*nWidth + idX; 
	int jj = j*nWidth + idY; 


	localImage[ii*nWidth+jj] = pInput[ii*nWidth+jj];

	barrier(CLK_LOCAL_MEM_FENCE);


    const int xOut = ii;
    const int yOut = jj;
    
    const int xInTopLeft = xOut;
    const int yInTopLeft = yOut;
    
    float sum = 0;
    for (int r = 0; r < nFilterWidth; r++)
    {
        const int idxFtmp = r * nFilterWidth;
        
        const int yIn = yInTopLeft + r;
        const int idxIntmp = yIn * nInWidth + xInTopLeft;
        
        for (int c = 0; c < nFilterWidth; c++)
        {
            const int idxF  = idxFtmp  + c;
            const int idxIn = idxIntmp + c;
            sum += pFilter[idxF]*localImage[idxIn];
        }
    }
    const int idxOut = yOut * nWidth + xOut;
    pOutput[idxOut] = sum;

}

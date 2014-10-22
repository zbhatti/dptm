#include "dpKernelFactory.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpSquareArray.hpp"
#include "dpConvolution.hpp"
#include "dpComputation.hpp"
#include "dpMemory.hpp"
#include "dpMatrixTranspose.hpp"
#include "dpVectorAdd.hpp"
#include "dpLUDecomposition.hpp"
#include "dpNBody.hpp"
#include "dpFWT.hpp"
#include "dpFloydWarshall.hpp"
#include "dpFluidSimulation.hpp"
#include "dpArray3dAverage.hpp"
#include "dpMonteCarloAsian.hpp"
#include "dpOxxxxx.hpp"
#include "dpReduction.hpp"
#include "dpEmpty.hpp"
#include "dpNoMemory.hpp"
//#include "dpCudaFFT.hpp" //excluding until fixed
//#include "dpFFT.hpp" //excluding until fixed
#include "dpCudaSquareArray.hpp"
#include "dpCudaVectorAdd.hpp"
#include "dpCudaMemory.hpp"
#include "dpCudaComputation.hpp"
#include "dpCudaMatrixMultiplication.hpp"
#include "dpCudaMatrixTranspose.hpp"
#include "dpCudaEmpty.hpp"
#include "dpCudaNoMemory.hpp"
#include "dpCudaOxxxxx.hpp"

#ifndef __dpKernelFactory_H_INCLUDED__
#define __dpKernelFactory_H_INCLUDED__

dpKernel* dpKernelFactory::makeTask(std::string name, cl_context context, cl_command_queue queue){	
		
		if (!name.compare("Computation"))
      return new dpComputation(context, queue);
		
    if (!name.compare("SquareArray"))
      return new dpSquareArray(context, queue);
			
		if (!name.compare("MatrixMultiplication"))
			return new dpMatrixMultiplication(context, queue);
			
		/*
		if (!name.compare("FFT"))
			return new dpFFT(context, queue);
		*/
		
		if (!name.compare("Memory"))
			return new dpMemory(context, queue);
		
		if (!name.compare("NoMemory"))
			return new dpNoMemory(context, queue);
		
		if (!name.compare("Empty"))
			return new dpEmpty(context, queue);
		
		if (!name.compare("FloydWarshall"))
			return new dpFloydWarshall(context, queue);
			
		if (!name.compare("FluidSimulation"))
			return new dpFluidSimulation(context, queue);
		
		if (!name.compare("MonteCarloAsian"))
			return new dpMonteCarloAsian(context, queue);
		
		if (!name.compare("Reduction"))
			return new dpReduction(context, queue);
		
		if (!name.compare("LUDecomposition"))
			return new dpLUDecomposition(context,queue);
		
		if (!name.compare("VectorAdd"))
			return new dpVectorAdd(context, queue);
		
		if (!name.compare("MatrixTranspose"))
			return new dpMatrixTranspose(context, queue);
		
		if (!name.compare("Convolution"))
			return new dpConvolution(context,queue);
		
		if (!name.compare("NBody"))
			return new dpNBody(context,queue);
		
		if (!name.compare("Oxxxxx"))
			return new dpOxxxxx(context,queue);
		
		if(!name.compare("Array3dAverage"))
			return new dpArray3dAverage(context,queue);
		
		/*
		if(!name.compare("CudaFFT"))
			return new dpCudaFFT(context,queue);
		*/
		
		if(!name.compare("CudaSquareArray"))
			return new dpCudaSquareArray(context,queue);
			
		if(!name.compare("CudaVectorAdd"))
			return new dpCudaVectorAdd(context,queue);
		
		if(!name.compare("CudaMatrixMultiplication"))
			return new dpCudaMatrixMultiplication(context,queue);
			
		if(!name.compare("CudaMatrixTranspose"))
			return new dpCudaMatrixTranspose(context,queue);
			
		if (!name.compare("CudaNoMemory"))
			return new dpCudaNoMemory(context, queue);
		
		if (!name.compare("CudaEmpty"))
			return new dpCudaEmpty(context, queue);
			
		if (!name.compare("CudaMemory"))
			return new dpCudaMemory(context, queue);
			
		if (!name.compare("CudaComputation"))
			return new dpCudaComputation(context, queue);
			
		if (!name.compare("CudaOxxxxx"))
			return new dpCudaOxxxxx(context, queue);
			
		else	//need better return case here
			return new dpSquareArray(context, queue);
 }

#endif
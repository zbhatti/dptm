#include "dpKernelFactory.hpp"
#include "dpFFT.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpSquareArray.hpp"
#include "dpRotateImage.hpp"
#include "dpConvolution.hpp"
#include "dpMersenneTwister.hpp"
#include "dpMatrixTranspose.hpp"
#include "dpVectorAdd.hpp"
#include "dpLUDecomposition.hpp"
#include "dpNBody.hpp"
#include "dpFWT.hpp"
#include "dpFloydWarshall.hpp"
#include "dpFluidSimulation.hpp"
#include "dpArray3dAverage.hpp"

#ifndef __dpKernelFactory_H_INCLUDED__
#define __dpKernelFactory_H_INCLUDED__

dpKernel* dpKernelFactory::makeTask(std::string name, cl_context context, cl_command_queue queue) {
    if (!name.compare("SquareArray"))
      return new dpSquareArray(context, queue);
			
		if (!name.compare("MatrixMultiplication"))
			return new dpMatrixMultiplication(context, queue);
			
		if (!name.compare("RotateImage"))
			return new dpRotateImage(context, queue);
		
		if (!name.compare("FFT"))
			return new dpFFT(context, queue);
		
		if (!name.compare("FWT"))
			return new dpFWT(context, queue);
		
		if (!name.compare("FloydWarshall"))
			return new dpFloydWarshall(context, queue);
			
		if (!name.compare("FluidSimulation"))
			return new dpFluidSimulation(context, queue);
		
		if (!name.compare("LUDecomposition"))
			return new dpLUDecomposition(context,queue);
		
		if (!name.compare("VectorAdd"))
			return new dpVectorAdd(context, queue);
		
		if (!name.compare("MatrixTranspose"))
			return new dpMatrixTranspose(context, queue);
		
		if (!name.compare("MersenneTwister"))
			return new dpMersenneTwister(context,queue);
		
		if (!name.compare("Convolution"))
			return new dpConvolution(context,queue);
		
		if (!name.compare("NBody"))
			return new dpNBody(context,queue);
		
		if(!name.compare("Array3dAverage"))
			return new dpArray3dAverage(context,queue);
		
		else	//need better return case here
			return new dpSquareArray(context, queue);
 }

#endif
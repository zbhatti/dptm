#include "dpKernelFactory.hpp"

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
		
		else	//need better return case here
			return new dpRotateImage(context, queue);
 }

#endif
#include "dpKernel.hpp"
#include "dpFFT.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpSquareArray.hpp"
#include "dpRotateImage.hpp"
#include <string>

class dpKernelFactory{
  
	public:
		dpKernel* makeTask(std::string, cl_context, cl_command_queue);
};
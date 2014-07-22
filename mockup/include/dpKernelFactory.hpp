#include "dpKernel.hpp"
#include <string>

class dpKernelFactory{
  
	public:
		dpKernel* makeTask(std::string, cl_context, cl_command_queue);
};
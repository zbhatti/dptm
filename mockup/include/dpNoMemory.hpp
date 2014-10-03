#include "dpKernel.hpp"
#ifndef __dpNoMemory_H_INCLUDED__
#define __dpNoMemory_H_INCLUDED__

class dpNoMemory: public dpKernel{
	public:
		dpNoMemory(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
};

#endif
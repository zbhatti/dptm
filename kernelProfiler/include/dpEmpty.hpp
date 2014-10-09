#include "dpKernel.hpp"
#ifndef __dpEmpty_H_INCLUDED__
#define __dpEmpty_H_INCLUDED__

class dpEmpty: public dpKernel{
	public:
		dpEmpty(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
};

#endif
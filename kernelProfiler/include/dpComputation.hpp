#include "dpKernel.hpp"
#ifndef __dpComputation_H_INCLUDED__
#define __dpComputation_H_INCLUDED__

class dpComputation: public dpKernel{

	float a, b;

	public:
		dpComputation(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif
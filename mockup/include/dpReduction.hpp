
/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "dpKernel.hpp"
#define VECTOR_SIZE 4
#define MULTIPLY  2  //Require because of extra addition before loading to local memory
#include <string.h>
#include <malloc.h>

#ifndef __dpReduction_H_INCLUDED__
#define __dpReduction_H_INCLUDED__

class dpReduction: public dpKernel{

	cl_uint length;                 /**< length of the input array */
	int numBlocks;                  /**< Number of groups */
	cl_uint *input;                 /**< Input array */
	cl_uint *outputPtr;             /**< Output array */
	cl_uint output;                 /**< Output result */
	cl_mem inputBuffer;             /**< CL memory buffer */
	cl_mem outputBuffer;             /**< CL memory buffer */
	
	public:
		dpReduction(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute(); 
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
		template<typename T> int fillRandom(T *,const int,const int,const T,const T);
};

#endif
#include "dpKernel.hpp"
#ifndef __dpFloydWarshall_H_INCLUDED__
#define __dpFloydWarshall_H_INCLUDED__

class dpFloydWarshall: public dpKernel{
		cl_int         	numNodes;  /**< Number of nodes in the graph */
		cl_uint        	*pathDistanceMatrix;  /**< path distance array */
		cl_uint         *pathMatrix;  /**< path arry */
		cl_mem        	pathDistanceBuffer; /**< CL path distance memory buffer */
		cl_mem          pathBuffer; /**< CL path memory buffer */
		cl_uint					blockSize;      /**< use local memory of size blockSize x blockSize */
		
	public:
		dpFloydWarshall(cl_context, cl_command_queue);
		void init(int,int,int);
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(cl_uint*, int, int);
};

#endif
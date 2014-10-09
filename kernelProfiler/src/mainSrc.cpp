#include <stdio.h>
#include "dpClient.hpp"
#define REPEAT 10
#define MAXMB 1024
#define MINMB 64
#define INC 64
#define RUNFILE cliList[i]->runTasks(); cliList[i]->printFile();
#define RUNSCREEN //cliList[i]->runTasks(); cliList[i]->printScreen();

int main (int argc, const char* argv[]) {
		//take platform and device argument:
		dpClient nvidiaTesla(0,0); 
		dpClient nvidia780(0,1);
		dpClient amdHawaii(1,0);
		dpClient amdCPU(1,1); 
		dpClient intelCPU(2,0);
		dpClient intelPhi(2,1);
		dpClient cudaTesla(3,0);
		dpClient cuda780(3,1);
		dpClient* cliList[8] = {&nvidiaTesla, &nvidia780, &amdHawaii, &amdCPU, &intelCPU, &intelPhi, &cudaTesla, &cuda780};

	//take task scan argument:
	for (int r=0; r<REPEAT; r++){
		for (int mb=MINMB; mb<=MAXMB; mb=mb+INC){
			for (unsigned int i=0; i<sizeof(cliList)/sizeof(cliList[0]); i++){
				fprintf(stderr, "\n\n########################\n%s-%s @ %dMiB\n########################\n\n", cliList[i]->getPlat(),cliList[i]->getDev(), mb);
				
				//CUDA Devices
				if ( !strcmp(cliList[i]->getType(),"CUDA")){

					fprintf(stderr,"CudaSquareArray, %d MB\n",mb);
					cliList[i]->addTask("CudaSquareArray",1,1,1,mb);
					RUNFILE
					RUNSCREEN
/*					
					fprintf(stderr,"CudaVectorAdd, %d MB\n",mb);
					cliList[i]->addTask("CudaVectorAdd",1,1,1,mb);
					RUNFILE
					RUNSCREEN
	
					fprintf(stderr,"CudaMatrixMultiplication, %d MB\n",mb);
					cliList[i]->addTask("CudaMatrixMultiplication",1,1,1,mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"CudaMatrixTranspose, %d MB\n",mb);
					cliList[i]->addTask("CudaMatrixTranspose",1,1,1,mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"CudaEmpty, %d MB\n", mb);
					cliList[i]->addTask("CudaEmpty",1,1,1,mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"CudaNoMemory, %d MB\n", mb);
					cliList[i]->addTask("CudaNoMemory",1,1,1,mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"CudaMemory, %d MB\n", mb);
					cliList[i]->addTask("CudaMemory",1,1,1,mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"CudaComputation, %d MB\n", mb);
					cliList[i]->addTask("CudaComputation",1,1,1,mb);
					RUNFILE
					RUNSCREEN
*/					
				}
				
				//OpenCL
				else{
/*
					fprintf(stderr,"Array3dAverage, %d MB\n",mb);
					cliList[i]->addWGScan("Array3dAverage",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"Convolution, %d MB\n",mb);
					cliList[i]->addWGScan("Convolution",mb);
					RUNFILE
					RUNSCREEN

					fprintf(stderr,"Computation, %d MB\n",mb);
					cliList[i]->addWGScan("Computation",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"Empty, %d MB\n",mb);
					cliList[i]->addWGScan("Empty",mb);
					RUNFILE
					RUNSCREEN

					fprintf(stderr,"FloydWarshall, %d MB\n",mb); 
					cliList[i]->addWGScan("FloydWarshall",mb);  
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"FluidSimulation, %d MB\n",mb);
					cliList[i]->addWGScan("FluidSimulation",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"FWT, %d MB\n",mb);
					cliList[i]->addWGScan("FWT",mb);
					RUNFILE
					RUNSCREEN
				
					fprintf(stderr,"LUDecomposition, %d MB\n",mb);
					cliList[i]->addWGScan("LUDecomposition",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"MatrixMultiplication, %d MB\n",mb);
					cliList[i]->addWGScan("MatrixMultiplication",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"MatrixTranspose, %d MB\n",mb);
					cliList[i]->addWGScan("MatrixTranspose",mb);
					RUNFILE
					RUNSCREEN
			
					fprintf(stderr,"Memory, %d MB\n",mb);
					cliList[i]->addWGScan("Memory",mb);
					RUNFILE
					RUNSCREEN
				
					fprintf(stderr,"MonteCarloAsian, %d MB\n",mb);
					cliList[i]->addWGScan("MonteCarloAsian",mb);
					RUNFILE
					RUNSCREEN
					
					fprintf(stderr,"NoMemory, %d MB\n",mb);
					cliList[i]->addWGScan("NoMemory",mb);
					RUNFILE
					RUNSCREEN

					//extremely slow kernel
					fprintf(stderr,"NBody, %d MB\n",mb);
					cliList[i]->addWGScan("NBody",mb);
					RUNFILE
					//RUNSCREEN
					
					
					fprintf(stderr,"Reduction, %d MB\n",mb);
					cliList[i]->addWGScan("Reduction",mb);
					RUNFILE
					RUNSCREEN
*/			
					fprintf(stderr,"SquareArray, %d MB\n",mb);
					cliList[i]->addWGScan("SquareArray",mb);
					RUNFILE
					RUNSCREEN
/*					
					fprintf(stderr,"VectorAdd, %d MB\n",mb);
					cliList[i]->addWGScan("VectorAdd",mb);
					RUNFILE
					RUNSCREEN
*/
				}
				
			}
		}
	}
	
	return 0;
}


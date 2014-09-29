#include <stdio.h>
#include "dpClient.hpp"
#define REPEAT 10
#define MAXMB 44
#define MINMB 4
#define INC 4
#define NDEVICES 8
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
		dpClient* cliList[NDEVICES] = {&nvidiaTesla, &nvidia780, &amdHawaii, &amdCPU, &intelCPU, &intelPhi, &cudaTesla, &cuda780}; 
	
	//take task scan argument:
	for (int r=0; r<REPEAT; r++){
		for (int mb=MINMB; mb<=MAXMB; mb=mb+INC){
			for (int i=0; i<6; i++){
				fprintf(stderr, "\n\n#######################\n%s-%s\n#######################\n\n", cliList[i]->getPlat(),cliList[i]->getDev());
				
				//CUDA Devices
				if (i>=6){
					
					fprintf(stderr,"CudaSquareArray, %d MB\n",mb);
					cliList[i]->addTask("CudaSquareArray",1,1,1,mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"CudaVectorAdd, %d MB\n",mb);
					cliList[i]->addTask("CudaVectorAdd",1,1,1,mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"CudaMatrixMultiplication, %d MB\n",mb);
					cliList[i]->addTask("CudaMatrixMultiplication",1,1,1,mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"CudaMatrixTranspose, %d MB\n",mb);
					cliList[i]->addTask("CudaMatrixTranspose",1,1,1,mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
				}
				
				//OpenCL
				/*
				else{ 
					//intel platform crashes
					fprintf(stderr,"Array3dAverage, %d MB\n",mb);
					cliList[i]->addWGScan("Array3dAverage",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();

					fprintf(stderr,"Convolution, %d MB\n",mb);
					cliList[i]->addWGScan("Convolution",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"FloydWarshall, %d MB\n",mb); 
					cliList[i]->addWGScan("FloydWarshall",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					*/
					
					fprintf(stderr,"FluidSimulation, %d MB\n",mb);
					cliList[i]->addWGScan("FluidSimulation",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					/*
					fprintf(stderr,"FWT, %d MB\n",mb);
					cliList[i]->addWGScan("FWT",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
				
					fprintf(stderr,"LUDecomposition, %d MB\n",mb);
					cliList[i]->addWGScan("LUDecomposition",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"MatrixMultiplication, %d MB\n",mb);
					cliList[i]->addWGScan("MatrixMultiplication",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"MatrixTranspose, %d MB\n",mb);
					cliList[i]->addWGScan("MatrixTranspose",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"MonteCarloAsian, %d MB\n",mb);
					cliList[i]->addWGScan("MonteCarloAsian",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					//extremely slow kernel
					fprintf(stderr,"NBody, %d MB\n",mb);
					cliList[i]->addWGScan("NBody",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"Reduction, %d MB\n",mb);
					cliList[i]->addWGScan("Reduction",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"SquareArray, %d MB\n",mb);
					cliList[i]->addWGScan("SquareArray",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
					
					fprintf(stderr,"VectorAdd, %d MB\n",mb);
					cliList[i]->addWGScan("VectorAdd",mb);
					cliList[i]->runTasks();
					cliList[i]->printFile();
					//cliList[i]->printScreen();
				}*/
				
			}
		}
	}
	
	return 0;
}


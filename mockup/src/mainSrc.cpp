#include <stdio.h>
#include "dpClient.hpp"
#define REPEAT 10
#define MAXMB 44
#define MINMB 26
int main (int argc, const char* argv[]) {
	
		//take platform and device argument:
		dpClient nvidiaTesla(0,0);
		dpClient amdHawaii(1,0);
		dpClient amdCPU(1,1);
		dpClient intelCPU(2,0);
		dpClient intelPhi(2,1);
		dpClient* cliList[5] = {&nvidiaTesla, &amdHawaii, &amdCPU, &intelCPU, &intelPhi}; 
	
	//take task scan argument:
	for (int r=0; r<REPEAT; r++){
		for (int mb=MINMB; mb<=MAXMB; mb=mb+2){
			for (int i=0; i<5; i++){
			fprintf(stderr, "\n\n\n\n%s\n\n", cliList[i]->getDev());
				
				/*
				can't get every 4th integer
				fprintf(stderr,"Array3dAverage, %d MB\n",mb);
				cliList[i]->addWGScan("Array3dAverage",mb);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printScreen();
				*/
				
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
				
				/*
				crashes horrendously on NVIDIA
				fprintf(stderr,"FluidSimulation, %d MB\n",mb);
				cliList[i]->addWGScan("FluidSimulation",mb);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printScreen();
				*/
				
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
				
				/*
				extremely slow kernel
				fprintf(stderr,"NBody, %d MB\n",mb);
				cliList[i]->addWGScan("NBody",mb);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printScreen();
				*/
				
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
				
			}
		}
	}
	
	return 0;
}


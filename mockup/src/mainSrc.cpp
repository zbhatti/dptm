#include <stdio.h>
#include "dpClient.hpp"
 
int main (int argc, const char* argv[]) {
	
	//take platform and device argument:
	
		dpClient cli1(0,0);
		dpClient cli2(1,0);
		dpClient cli3(1,1);
		dpClient cli4(2,0);
		dpClient cli5(2,1);
		dpClient* cliList[5] = {&cli1, &cli2, &cli3, &cli4, &cli5}; 
	
	//take task scan argument:
	for (int r=0;r<10;r++){
		for (int j=8; j<=64; j=j+4){ 
			for (int i=0;i<5;i++){
			fprintf(stderr, "Client:%d\n",i);
				
				fprintf(stderr,"Array3dAverage, %d MB\n",j);
				cliList[i]->addWGScan("Array3dAverage",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"Convolution, %d MB\n",j);
				cliList[i]->addWGScan("Convolution",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"FloydWarshall, %d MB\n",j); 
				cliList[i]->addWGScan("FloydWarshall",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				/*
				fprintf(stderr,"FluidSimulation, %d MB\n",j);
				cliList[i]->addWGScan("FluidSimulation",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				*/
				
				fprintf(stderr,"FWT, %d MB\n",j);
				cliList[i]->addWGScan("FWT",j);
				cliList[i]->runTasks();
				//cliList[i]->printFile();
				cliList[i]->printTimes();
				
				fprintf(stderr,"LUDecomposition, %d MB\n",j);
				cliList[i]->addWGScan("LUDecomposition",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"MatrixMultiplication, %d MB\n",j);
				cliList[i]->addWGScan("MatrixMultiplication",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"MatrixTranspose, %d MB\n",j);
				cliList[i]->addWGScan("MatrixTranspose",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"MonteCarloAsian, %d MB\n",j);
				cliList[i]->addWGScan("MonteCarloAsian",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"NBody, %d MB\n",j);
				cliList[i]->addWGScan("NBody",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"Reduction, %d MB\n",j);
				cliList[i]->addWGScan("Reduction",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"SquareArray, %d MB\n",j);
				cliList[i]->addWGScan("SquareArray",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
				fprintf(stderr,"VectorAdd, %d MB\n",j);
				cliList[i]->addWGScan("VectorAdd",j);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				//cliList[i]->printTimes();
				
			}
		}
	}
	
	return 0;
}


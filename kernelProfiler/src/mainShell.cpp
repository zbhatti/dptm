#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include "dpClient.hpp"
using namespace std;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

int main (int argc, const char* argv[]) {
	string inputString;
	int platform;
	int device;
	int mbStart;
	int mbEnd;
	int mbIncrement;
	int repeat;
	int printScreen = 1;
	int printFile = 0;
	std::vector<std::string> kernels;
	std::vector<dpClient> clientList;
	
	
	//add Devices Loop
	getline(cin,inputString);
	printf("%s\n", inputString.c_str());
	for(getline(cin,inputString); inputString[0] != '#'; getline(cin,inputString) ){
		std::vector<std::string> params = split(inputString, ',');
		stringstream(params.at(0)) >> platform;
		stringstream(params.at(1)) >> device;
		dpClient client(platform, device);
		clientList.push_back(client);
	}
	
	//populate Kernels Loop
	printf("%s\n", inputString.c_str());
	for(getline(cin,inputString); inputString[0] != '#'; getline(cin,inputString) ){
		kernels.push_back(inputString);
	}
	
	printf("%s\n", inputString.c_str());
	
	if (!inputString.compare("#MBSTART:")){
		getline(cin,inputString);
		stringstream(inputString) >> mbStart;
		getline(cin,inputString);
	}
	
	if (!inputString.compare("#MBEND:")){
		getline(cin,inputString);
		stringstream(inputString) >> mbEnd;
		getline(cin,inputString);
	}
	
	if (!inputString.compare("#MBINCREMENT:")){
		getline(cin,inputString);
		stringstream(inputString) >> mbIncrement;
		getline(cin,inputString);
	}
	
	if (!inputString.compare("#REPEAT:")){
		getline(cin,inputString);
		stringstream(inputString) >> repeat;
		getline(cin,inputString);
	}
	
	if (!inputString.compare("#PRINTSCREEN:")){
		getline(cin,inputString);
		stringstream(inputString) >> printScreen;
		getline(cin,inputString);
	}
	
	if (!inputString.compare("#PRINTFILE:")){
		getline(cin,inputString);
		stringstream(inputString) >> printFile;
		getline(cin,inputString);
	}
	
	for(int r = 0; r < repeat; r++){
		for(int mb = mbStart; mb < mbEnd; mb += mbIncrement){
			for(unsigned int c=0; c < clientList.size(); c++){
				fprintf(stderr,"\n\n########################\n%s-%s @ %dMiB\n########################\n\n", clientList.at(c).getPlat(), clientList.at(c).getDev(), mb);
				for(unsigned int k=0; k < kernels.size(); k++){
					
					if (!strcmp(clientList.at(c).getType(),"CUDA") && !kernels.at(k).find("Cuda")){ //Cuda client and cuda kernel
						fprintf(stderr,"adding CUDA: %s\n",kernels.at(k).c_str());
						clientList.at(c).addTask(kernels.at(k),1,1,1,mb);
					}
					
					
					
					if(!strcmp(clientList.at(c).getType(),"OpenCL")  && kernels.at(k).find("Cuda")){ //OpenCL client and NOT cuda kernel
						fprintf(stderr,"adding OpenCL: %s\n",kernels.at(k).c_str());
						clientList.at(c).addWGScan(kernels.at(k),mb);
					}
				}
				clientList.at(c).runTasks();
				if(printScreen)
					clientList.at(c).printScreen();
				if(printFile)
					clientList.at(c).printFile();
			}
		}
	}
}
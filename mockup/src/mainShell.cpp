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

//all values should have defaults
int main (int argc, const char* argv[]) {
	string inputString;
	
	int platform = 0;
	int device = 0;
	string kernel = "SquareArray";
	int MB = 4;
	int xLocal = 4;
	int yLocal = 4;
	int zLocal = 4;
	
	getline (cin,inputString);
	stringstream(inputString) >> platform;
	getline (cin,inputString);
	stringstream(inputString) >> device;
	dpClient client(platform, device);
	
	while(1){
		
		getline (cin,inputString);
		if (!inputString.compare("RUN")) //use string compare method
			break;
		
		std::vector<std::string> parameters = split(inputString, ',');
		
		kernel = parameters.at(0);
		stringstream(parameters.at(1)) >> MB;
		stringstream(parameters.at(2)) >> xLocal;
		stringstream(parameters.at(3)) >> yLocal;
		stringstream(parameters.at(4)) >> zLocal;
		fprintf(stderr,"adding: %s,%d,%d,%d,%d\n",kernel.c_str(),MB,xLocal,yLocal,zLocal);
		client.addTask(kernel,xLocal,yLocal,zLocal,MB);
	}

	client.runTasks();
	//client.printTimes();
	client.printFile();

}
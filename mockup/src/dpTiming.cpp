#include "dpTiming.hpp"
#include <sstream>
#include <iostream>

std::string dpTiming::getVariables(){
	std::stringstream ss;
	ss<<"device,";
	ss<<"kernel,";
	for (unsigned int i = 0; i < dataNames.size();i++){
		ss<<dataNames.at(i)<<",";
	}
	ss<<"xLocal,";
	ss<<"yLocal,";
	ss<<"zLocal,";
	ss<<"memoryCopyOut,";
	ss<<"plan,";
	ss<<"execute,";
	ss<<"memoryCopyIn,";
	ss<<"cleanUp,";
	return ss.str();
}

std::string dpTiming::getTimes(){
	std::stringstream ss;
	ss<<device<<",";
	ss<<name<<",";
	for (unsigned int i = 0; i < data.size();i++){
		ss<<data.at(i)<<",";
	}
	ss<<localSize[0]<<",";
	ss<<localSize[1]<<","; 
	ss<<localSize[2]<<",";
	ss<<memoryCopyOut<<",";
	ss<<plan<<",";
	ss<<execute<<",";
	ss<<memoryCopyIn<<",";
	ss<<cleanUp<<",";
	return ss.str();
}
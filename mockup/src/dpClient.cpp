#include "dpClient.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

//hardcoding device names and platforms for easier readability specific to AFSuper
void nameFixer(int platform, int device, char*plat, char*dev){
	if (platform == 0){
		strcpy(plat, "NvidiaRuntime");
		if (device == 0)
			strcpy(dev, "Tesla K20Xm");
		if (device == 1)
			strcpy(dev, "GTX 780");
	}
	
	if (platform == 1){
		strcpy(plat, "AmdRuntime");
		if (device == 0)
			strcpy(dev, "AMD Firepro W9100");
		if (device == 1)
			strcpy(dev, "Intel Xeon CPU E5-2695 v2 @ 2.40GHz");
	}
	
	if (platform == 2){
		strcpy(plat, "IntelRuntime");
		if (device == 0)
			strcpy(dev, "Intel Xeon CPU E5-2695 v2 @ 2.40GHz");
		if (device == 1)
			strcpy(dev, "Intel Xeon Phi coprocessor x100 family");
	}
	
		if (platform == 3){
		strcpy(plat, "CUDA");
		if (device == 0)
			strcpy(dev, "Tesla K20Xm");
		if (device == 1)
			strcpy(dev, "GTX 780");
	}
	
}

//set up context and queue on a device and retrieve 
//device information for other methods
dpClient::dpClient(int platform, int device){
	
	nameFixer(platform, device, platName, devName);
	fprintf(stderr,"On Platform %s\n", platName);
	fprintf(stderr,"using device %s\n", devName);
	//OpenCL
	if (platform != 3){
		unsigned int numDevices;
		int err;
		cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,0,0};
		clErrChk(clGetPlatformIDs(16, platform_ids, NULL));
		clErrChk(clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, 16, device_ids, &numDevices));
		clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, NULL));
		clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MaxComputeUnits), &MaxComputeUnits, NULL));
		clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(MaxWorkDim), &MaxWorkDim, NULL));
		clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(MaxMemAlloc), &MaxMemAlloc, NULL));
		
		props[1] = (cl_context_properties) platform_ids[platform];
		context = clCreateContext(props, 1, &device_ids[device], NULL, NULL, &err); 
		clErrChk(err);
		queue = clCreateCommandQueue( context, device_ids[device], 0, &err); 
		clErrChk(err);
	}
	
	
	//CUDA
	else{
		//CLIENT:
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		fprintf(stderr, "%s\n", properties.name);
		//fprintf(stderr,"Device %d has %d multiProcessors. \n",device, properties.multiProcessorCount);
		//fprintf(stderr,"Selected Device %d.\n",device);
		cudaSetDevice(device);
		MaxWorkGroupSize = 1;
		MaxMemAlloc = 100000;
	}

	
}


//Add kernel task:, xLocal=1, yLocal=1, zLocal=1, MB=8 if not passed to this function
void dpClient::addTask(std::string name, int xLocal, int yLocal, int zLocal, int MB){
	//check that the wg arguments do not exceed the maxworkgroupsize of a device 
	if(( xLocal>(int)MaxWorkDim[0]) ||
			(yLocal>(int)MaxWorkDim[1]) ||
			(zLocal>(int)MaxWorkDim[2]) ||
			(xLocal*yLocal*zLocal>(int)MaxWorkGroupSize)
		)
		xLocal=yLocal=zLocal = 8;
	
	taskList.push_back(kernelFactory.makeTask(name, context, queue));
	taskList.at(taskList.size()-1)->setup(MB,xLocal,yLocal,zLocal);
}

//Add scan over data size at a constant workgroupdimension, default xLocal=yLocal=zLocal=1
void dpClient::addMBScan(std::string name, int xLocal, int yLocal, int zLocal){
	//scan from 1MB to Max MB of single allocation for a kernel
	for (int i=0; pow(2,i) <= 8; i++)
		addTask(name, xLocal, yLocal, zLocal, pow(2,i) );
}

//Add scan over workgroup dimensions at a constant data size, default MB=8
void dpClient::addWGScan(std::string name, int MB){
	workGroupSpace workDim;
	int i,j,k;
	j =0;
	k =0;
	
	//make a temporary kernel to read work dimension from:
	workDim = kernelFactory.makeTask(name, context, queue)->workDimension;
	
	//add a 1D wg kernel
	if (workDim == ONE_D){
		for (i=0; pow(2,i)<=MaxWorkGroupSize; i++){
			addTask(name, pow(2,i),1,1, MB);
		}
	}
	
	//add a 2D wg kernel
	if (workDim == TWO_D){
		for(i=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; j++){
				addTask(name, pow(2,i), pow(2,j),1,MB);
			}
			j=0;
		}
	}
	
	//add a 3D wg kernel
	if (workDim == THREE_D){
		for(i=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; j++){
				for(k=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; k++){
					addTask(name, pow(2,i), pow(2,j), pow(2,k),MB);
				}
				k=0;
			}
			j=0;
		}
	}
	
}

//run a taskList. after finishing, clear the taskList
void dpClient::runTasks(){
	dpTiming timeTmp;
	int err;
	
	for (unsigned int i =0; i <taskList.size(); i++){
		
		timeTmp.name = taskList.at(i)->name;
		timeTmp.localSize = taskList.at(i)->getLocalSize();
		timeTmp.MB = taskList.at(i)->getMB();
		timeTmp.device = devName;
		
		//if block for enum to string:
		if (taskList.at(i)->workDimension == ONE_D)
			timeTmp.workDimension ="ONE_D";
		if (taskList.at(i)->workDimension == TWO_D)
			timeTmp.workDimension = "TWO_D";
		if (taskList.at(i)->workDimension == THREE_D) 
			timeTmp.workDimension = "THREE_D";
		
		
		gettimeofday(&start, NULL);
		taskList.at(i)->init();
		gettimeofday(&finish, NULL);
		timeTmp.init = timeDiff(start,finish);
		
		timeTmp.data= taskList.at(i)->dataParameters;
		timeTmp.dataNames= taskList.at(i)->dataNames;
		
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyOut();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyOut = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->plan();
		gettimeofday(&finish, NULL);
		timeTmp.plan = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		err = taskList.at(i)->execute();
		gettimeofday(&finish, NULL);

		//execution failed: set error values and continue testing
		if (err<0){
			taskList.at(i)->memoryCopyIn();
			taskList.at(i)->cleanUp();
			timeTmp.execute = -1;
			timeTmp.memoryCopyIn = -1;
			timeTmp.cleanUp = -1;
			timeList.push_back(timeTmp);
			continue;
		}
		else
			timeTmp.execute = timeDiff(start,finish);
			
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyIn();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyIn = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->cleanUp();
		gettimeofday(&finish, NULL);
		timeTmp.cleanUp = timeDiff(start,finish);
		
		timeList.push_back(timeTmp);
	}
	
	taskList.clear();
}

//helper function that returns time difference
float dpClient::timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000.0 + finish.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec))/(1000.0);
}

//print times to stdout. this function clears the timeList when done
void dpClient::printScreen(){
	printf("%s\n",timeList.at(0).getVariables().c_str());
	for (unsigned int i = 0; i < timeList.size(); i++){
		if(i>0){
			if ( timeList.at(i).name.compare(timeList.at(i-1).name) )
				printf("%s\n",timeList.at(i).getVariables().c_str());
		}
		printf("%s\n", timeList.at(i).getTimes().c_str() );
	}
	timeList.clear();
}

//print times to files. this function clears the timeList when done
void dpClient::printFile(){
	std::ofstream fileOut;
	std::ifstream fileIn;
	char tmp[256];
	std::stringstream tmpString;
	strcpy(tmp, "analysis/");
	strcat(tmp, platName);
	strcat(tmp," - ");
	strcat(tmp, devName);
	mkdir(tmp, 0777);
	
	for (unsigned int i=0; i<timeList.size(); i++){
		//get filename to save to at ./device/kernelname.log
		tmpString << tmp << "/" << timeList.at(i).name.c_str() << timeList.at(i).MB << ".log";
		fileOut.open(tmpString.str().c_str(), std::ofstream::app );
		fileIn.open(tmpString.str().c_str() );
		
		//if the file is empty, add the variable names first
		if (isEmpty(fileIn))
			fileOut << timeList.at(i).getVariables().c_str() << "\n";

		fileOut << timeList.at(i).getTimes().c_str() << "\n";
		fileOut.close();
		fileIn.close();
		tmpString.str("");
	}
	
	timeList.clear();
}

//helping function to check if a file is empty
//http://stackoverflow.com/questions/2390912/checking-for-an-empty-file-in-c
bool dpClient::isEmpty(std::ifstream& pFile){
    return pFile.peek() == std::ifstream::traits_type::eof();
}




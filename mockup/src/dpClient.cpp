#include "dpClient.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }


//set up context and queue on a device and retrieve valuable
//device information for other methods
dpClient::dpClient(int platform, int device){
	unsigned int numDevices;
	char name[256];
	int err;
	cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,0,0};
	clErrChk(clGetPlatformIDs(16, platform_ids, NULL));
	clErrChk(clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, 16, device_ids, &numDevices));
	clErrChk(clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, sizeof(name), name, NULL));
	fprintf(stderr,"On Platform %s\n", name);
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_NAME, sizeof(name), name, NULL));
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, NULL));
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MaxComputeUnits), &MaxComputeUnits, NULL));
	fprintf(stderr,"using device %s\n", name);
	props[1] = (cl_context_properties) platform_ids[platform];
	context = clCreateContext(props, 1, &device_ids[device], NULL, NULL, &err); clErrChk(err);
	queue = clCreateCommandQueue( context, device_ids[device], 0, &err); clErrChk(err);
}


void dpClient::runKernels(){

	//workgroups must be smaller than data dimensions
	dpMatrixMultiplication MM1(context, queue);
	dpMatrixMultiplication MM2(context, queue);
	dpMatrixMultiplication MM3(context, queue);

	dpSquareArray square1(context, queue);
	dpSquareArray square2(context, queue);
	dpSquareArray square3(context, queue);
	
	dpRotateImage rot1(context, queue);
	dpRotateImage rot2(context, queue);
	dpRotateImage rot3(context, queue);
	
	dpFFT fft1(context, queue);
	dpFFT fft2(context, queue);
	dpFFT fft3(context, queue);
	
	taskList.push_back(&MM1);
	taskList.push_back(&MM2);
	taskList.push_back(&MM3);
	taskList.push_back(&square1);
	taskList.push_back(&square2);
	taskList.push_back(&square3);
	taskList.push_back(&rot1);
	taskList.push_back(&rot2);
	taskList.push_back(&rot3);
	taskList.push_back(&fft1);
	taskList.push_back(&fft2);
	taskList.push_back(&fft3);
	
	dpTiming timeTmp; //timeTmp needs to check what type it is so it can store arguments like N.M.P, WorkGroupSize, etc
	
	for (unsigned int i =0; i <taskList.size(); i++){
		if (taskList.at(i)->workDimension == ONE_D){
			//loop this over workgroup dimensions
			taskList.at(i)->init(MaxWorkGroupSize,1,1);
		}

		if (taskList.at(i)->workDimension == TWO_D){
			taskList.at(i)->init(8,32,1);
		}
		
		if (taskList.at(i)->workDimension == THREE_D){
			taskList.at(i)->init(16,4,1);
		}

		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyOut();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyOut = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->plan();
		gettimeofday(&finish, NULL);
		timeTmp.plan = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->execute();
		gettimeofday(&finish, NULL);
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
	
}
 
float dpClient::timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000.0 + finish.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec))/(1000.0);
}


//print times, probably change to export the timeList instance
void dpClient::printTimes(){
	for (unsigned int i = 0; i < timeList.size(); i++){
		printf("%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\n", 
			timeList.at(i).memoryCopyOut,
			timeList.at(i).plan,
			timeList.at(i).execute, 
			timeList.at(i).memoryCopyIn,
			timeList.at(i).cleanUp);
	}
}

int main(){
	dpClient cli1(0,0);
	cli1.runKernels();
	cli1.printTimes();
	
	dpClient cli2(1,0);
	cli2.runKernels();
	cli2.printTimes();
	
	dpClient cli3(1,1);
	cli3.runKernels();
	cli3.printTimes();
	
	dpClient cli4(2,0);
	cli4.runKernels();
	cli4.printTimes();
	
	dpClient cli5(2,1);
	cli5.runKernels();
	cli5.printTimes();
	
	/*
	int j=0;
	for(r=0; r<15; r++){
		for(i=0; pow(2,i)*pow(2,j)<=1024;i++){
			for(j=0; pow(2,i)*pow(2,j)<=1024;j++){
				cli1.matrixMultiplication(2048,2048,2048,pow(2,i),pow(2,j));
			}
			j=0;
		}
	}*/
	
	/*
	cli1.FFT(8);
	cli1.FFT(8192);
	for(r=0; r<2; r++){
		for(i=0; pow(2,i)*pow(2,j)<=cli1.MaxWGSize();i++){
			for(j=0; pow(2,i)*pow(2,j)<=cli1.MaxWGSize();j++){
				cli1.rotateImage(pow(2,i),pow(2,j));
			}
			j=0;
		}
	}
	
	for(r=0; r<2; r++){
		for(i=0; pow(2,i)*pow(2,j)<=cli2.MaxWGSize();i++){
			for(j=0; pow(2,i)*pow(2,j)<=cli2.MaxWGSize();j++){
				cli2.rotateImage(pow(2,i),pow(2,j));
			}
			j=0;
		}
	}
	*/
	return 0;
}








gcc src/clDeviceDetails.c -oDeviceDetails  -I/opt/intel/include -lOpenCL
gcc src/clMemoryTest.c -oMemoryTest -I/opt/intel/include -lOpenCL
gcc src/clListDevices.c -oListDevices -I/opt/intel/include -lOpenCL

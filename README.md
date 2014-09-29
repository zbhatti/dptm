DPTM
=========

DPTM stands for Data Parallel Task Manager. Its objectives are to:

  - Utilize parallel processors with minimum changes to existing serial code
  - Manage and schedule tasks to insure full utilization of available hardware
  - Be compatible with parallel APIs like OpenCL, CUDA, MPSS and more

Specifically, as outlined by Dr. Amir Farbin:

>The goal is to establish infrastructure for hybrid multi-core processor (ie CPU) and many-core co-processor (eg a GPU) computing in complex applications consisting of large numbers ofalgorithms, without major redesign and reworking of legacy software. The vision is that Task Parallel (TP) applications running on CPU cores schedule large numbers of Data Parallel (DP) computing tasks on available many-core co-processors, maximally utilizing the all available hard-ware resources. The project will be executed in the context of offline and trigger reconstruction of Large Hadron Collider (LHC) collisions, but will certainly have a broader impact because it is also applicable to particle interaction simulation (ie Geant4), streaming Data Acquisition Systems, statistical analysis, and other scientific applications.


Folder Structure
-----------

The current version of this project has explored different smaller projects and this
is reflected in the folder layout:

* [aside] - standalone OpenCL programs used to test system stability
  * _clFFT_ - simpler version of ‘ClientStandAloneOpenCL’ found in ‘singleFFT’
  * _matrixMultiplication_ - used to test different openCL implementations of matrixMultiplication and determine library linking compatibility
  * _monteCarlo_ - unaltered monteCarlo example from Intel SDK
  * _squareArray_ - used to compare consisitency of squareArray implemented with OpenCL and CUDA
* [doc] - contains the UML class diagram for the mockup folder
* [mockup] - OpenCL kernel launcher with performance reports
  * _analysis_ 
    * _Platform - Device_ - contains raw data from dpClientSrc with printFile enabled
    * _results_
      * _Platform - Device_ - contains graphs for time(x,y,z,mb)
    * _optimal_ - contains curves for time(mb) for each kernel on each device
  * _include_ - header files
  * _obj_ - object files
  * _src_ - source files
* [prototype] - cuFFT implementation with a prototype task manager
  * _client.cu_ - makes requests to dptm.cu using IPC
  * *wrapper_FFT.cu* - launches kernel on command from dptm.cu
  * _dptm.cu_ - takes client.cu requests and combines them to give to wrapper_FFT.cu
* [singleFFT] -  FFTW, clFFT and cuFFT comparisons without the use of the task manager
  * _analysis_
    * _oroduruin_ - raw data from tests run on the local OSX computer
    * _AFSuper_ - raw data from tests run on the local Red Hat computer
  * _conclusion_ - contains test results in graphs from pyROOT scripts
* [utilities] - OpenCL programs used to fetch information about the current PC's hardware


Installation
--------------
Installation begins with cloning the git repository:

```sh
mkdir dptm
git github.com:zbhatti/dptm.git dptm
cd dptm
```
Currently, only Red Hat flavor systems have been tested: [RHL6Installation]

Devices Tested
--------------
* NVIDIA GTX 650
* NVIDIA GTX 780
* NVIDIA GTX Titan
* NVIDIA Tesla K20Xm
* AMD Firepro W9100
* Intel Xeon E5-2695
* Intel Xeon Phi x100

[aside]:https://github.com/zbhatti/dptm/tree/master/aside
[doc]:https://github.com/zbhatti/dptm/tree/master/doc
[mockup]:https://github.com/zbhatti/dptm/tree/master/mockup
[prototype]:https://github.com/zbhatti/dptm/tree/master/prototype
[singleFFT]:https://github.com/zbhatti/dptm/tree/master/singleFFT
[utilities]:https://github.com/zbhatti/dptm/tree/master/utilities
[RHL6Installation]:https://github.com/zbhatti/dptm/wiki/Setup-Scientific-Linux-6
[Gitdptm]:git@github.com:zbhatti/dptm.git


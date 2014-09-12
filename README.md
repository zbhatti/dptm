DPTM
=========

DPTM stands for Data Parallel Task Manager. Its objectives are to:

  - Utilize parallel processors with minimum changes to existing serial code
  - Manage and schedule tasks to insure full utilization of available hardware
  - Be compatible with parallel APIs like OpenCL, CUDA, MPSS and more

Specifically, as outlined by Dr. Amir Farbin:

>The goal is to establish infrastructure for hybrid multi-core processor (ie CPU) and many-core co-processor (eg a GPU) computing in complex applications consisting of large numbers ofalgorithms, without major redesign and reworking of legacy software. The vision is that Task Parallel (TP) applications running on CPU cores schedule large numbers of Data Parallel (DP) computing tasks on available many-core co-processors, maximally utilizing the all available hard-ware resources. The project will be executed in the context of offline and trigger reconstruction of Large Hadron Collider (LHC) collisions, but will certainly have a broader impact because it is also applicable to particle interaction simulation (ie Geant4), streaming Data Acquisition Systems, statistical analysis, and other scientific applications.



This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.  

Version
----

2.0

Folder Structure
-----------

The current version of this project has explored different smaller projects and this
is reflected in the folder layout:

* [aside] - standalone OpenCL programs used to test system stability
* [doc] - contains the UML class diagram for the mockup folder currently
* [mockup] - OpenCL kernel launcher with performance reports 
* [prototype] - cuFFT implementation with a prototype task manager
* [singleFFT] - FFTW, clFFT and cuFFT comparisons without the use of the task manager
* [utilities] - OpenCL programs used to fetch information about the current PC's hardware

Installation
--------------

```sh
git clone [git-repo-url] dillinger
cd dillinger
npm i -d
mkdir -p public/files/{md,html,pdf}
```

##### Configure Plugins. Instructions in following README.md files

* plugins/dropbox/README.md
* plugins/github/README.md
* plugins/googledrive/README.md

```sh
node app
```


License
----

MIT


**Free Software, Hell Yeah!**
[aside]:https://github.com/zbhatti/dptm/tree/master/aside
[doc]:https://github.com/zbhatti/dptm/tree/master/doc
[mockup]:https://github.com/zbhatti/dptm/tree/master/mockup
[prototype]:https://github.com/zbhatti/dptm/tree/master/prototype
[singleFFT]:https://github.com/zbhatti/dptm/tree/master/singleFFT
[utilities]:https://github.com/zbhatti/dptm/tree/master/utilities

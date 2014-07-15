#include "dpKernel.hpp"
#include "dpFFT.hpp"
#include "dpMatrixMultiplication.hpp"
#include "dpSquareArray.hpp"

class dpKernelFactory{
  dpKernelFactory() {};
  ~dpKernelFactory() {};

  dpKernel *BuildKernel(char *name, context, queue) {

    if (name == "FFT")
      return 

  }

}

# ParallelDefault
Parallel computation of sovereign default model

In this repository, we show GPU impletations of sovereign default model. It includes models of C++ with CUDA thrust, C++ with OpenACC, CUDA with Julia, and a newly released stdpar impletation with nvc++ compiler.

C++ allows standard impletations of model with CUDA and OpenACC, we tested performance of thrust library in CUDA and multiloops in OpenACC. Julia also provides a variety of methods to do parallel computation. We started with a synchornized distributed model. Then we include Julia's CUDA library and redesigned the model to be run with GPU. Julia provides helpful management of memory by automatically allocating pointers without specifying host and device, however this comes with a communication cost.

With the release of stdpar library from Nvidia's HPC SDK in August, we were able to run CUDA using standard C++ code with nvc++ compiler. The design was based on std library's tansform and for_each methods. Challenge arose in developing a data structure to allow stdpar to transfer memory and manipulate data automatically both in CPU and GPU. We designed a class that could correctly transfer memory to GPU, and allows manipulation of matrices with lambda functions in standard library functions.


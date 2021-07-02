# Parallel Computation of Sovereign Default Models
Parallel computation of sovereign default model

This paper discusses the parallel and efficient computation of macroeconomic models, with emphasis on solving sovereign default models. Our motivation is two-fold.

First, we aim to streamline complex numerical models in a parallel computation fashion. 

Second, we want to bypass the steep learning and implementation costs of languages like C++ CUDA (Compute Unified Device Architecture) in economic research. To this end, we propose a framework for efficient parallel computing with the modern language Julia. The paper offers detailed analysis of parallel computing, Julia-style acceleration tricks, and coding advice. The benchmark implementation in Julia with CUDA shows a substantial speed up over 1,000 times compared to standard Julia. We provide an accompanying Github repository with the codes and the benchmarks. 

In this repository, we include the model implementation and benchmarks. In the models folder are the GPU implementations of sovereign default model. It includes models of C++ with CUDA thrust, CUDA with Julia, and a newly released stdpar implementation with the nvc++ compiler. In the benchmark folders are the benchmark codes and results.


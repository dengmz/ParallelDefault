

### Part 4 Julia CUDA

#### Kernels

The design of kernels:

threadIdx, blockDim, blockIdx

The main difficulty in the default model is efficient calculation of expected values. Each random variable and conditional variable contributes to one extra loop in the calculation, as well as one extra dimension in the probability matrix. #explain the extra layer

We showcase different methods in expectation calculation, from high-level API to low-level kernels. We put special emphasis on the low level kernels. With Julia low-level style, the lines of code significantly decreases, and programming experience is improved by dynamic types and checked arithmetic [1]

##### Value of Default

The one-dimensional operations uses the standard stride method (check name), which...

```julia
#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = CUDA.pow(CUDA.exp((1-τ)*Y[i]),(1-α))/(1-α)
    end
    return
end
```

Julia CUDA provides another method of fused broadcast for the code to run on GPU, as implemented below:

[Add code for broadcast implementation]

Calculating expected value requires one round of calculation for each after state y given the before state iy. By design of the kernel, we can calculate E[f(y)|iy] in one thread located at (iy,y). The number of random variables usually exceeds two, we divide the case up and discuss in detail for each case.

Calculating value of default requires E[f(y)|iy]. The following kernel stores a temporary result for f(y)|iy in a Ny*Ny matrix, with one thread to calculate one grid point.

```Julia
#adding expected value to sumdef
function def_add(temp, P, β, V0, Vd0, ϕ, Ny)
    y = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
  
		if (iy <= Ny && y <= Ny)
    		temp[iy,y] = β* P[iy,y]* (ϕ* V0[y,1] + (1-ϕ)* Vd0[y])

end
return
```
Given closer inspection of f(x), the process can be equivalently calculated through loop fusion operations, where operation ".*" multiplies respective elements at same position in two matrices.

```julia
A = ϕ* V0[:,1] + (1-ϕ)* Vd0
temp = β* P .* CUDA.transpose(A) 
#equivalent to β* P[iy,y]* A'[iy,y] for each (iy,y)
```
$E[\phi V(y',0) + (1-\phi)V_d(y')|y(i_y))] = \sum iy=1...Ny \phi V(y',0) + (1-\phi)V_d(y')$. Sum up the temp matrix along the iy axis, this is done by reducing the temporary matrix along the second dimension.

```julia
sumdef += reduce(+, temp, dims=2)
```

CUDA reduce parallel operation is based on shuffle instructions [1], thus providing exchange of data between threads in a same threadblock, and eliminating the need for shared data or synchronizing cost. 

##### Value of Repayment

Value of repayment consumes the largest bulk of computation power and should be the priority of optimization. The computation cost comes from to an expected value calculation consisting four variables. This paper develops from existing implementation with CPU and Thrust in CUDA. For the CPU version, Julia-style linear algebra and loop fusion operations replace the standardized for-loops. In the CUDA implementation, division of kernels provide simple and efficient computation by reducing synchronizing cost.

Inspection of existing the Thrust code shows optimization area: the design provides efficient transition of CPU code to GPU code. However, the executions for each grid point on each thread is lengthy. The quicker threads will wait for the slower threads to finish computing, and same data will be calculated and stored on device for each thread, requiring extra device space and computation power.  A simple and efficient fix is to divide the value repayment calculation into components. Each component can be individually and parallelly calculated.

The first component of repayment calculation, $ \max_{b'} U(c) = \max_ U(c(b')])$. We designate one kernel for each parallelizable process. The utility calculation requires essentially a broadcast operation. On the surface a line of broadcast as below would be sufficient:

```
function U2(x)
    return (x>=0) * (x+0im)^(1-α) / (1-α)
end

@benchmark C2 = U2.(C)
```

However the code is largely deprecated. The utility function is non-linear and does not accept negative input. For broadcasting purpose, we transform x to x+0im to temporary allow negative float calculation with imaginary number, and set the value back to 0 by multiplying term (x>=0).

The broadcast itself does not provide the desired speed: #something with scalar array...

Two simple kernels divides the step with better performance and clarity. Apart from the easy copy and paste headers, the kernels achieve the same level of brevity. 

The division of utility function calculation significantly lowers lag from to synchronization cost. In addition, by introducing a device matrix to store cost C, the method reduces temporary allocation and release of memory in each thread.

```julia
#Calculate c
function vr_C(Ny,Nb,Y,B,Price0,P,C)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            C[iy,ib,b] = -Price0[iy,b]*B[b] + CUDA.exp(Y[iy]) + B[ib]
        end
    end
end
@benchmark @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
```
```julia
#Calculate U(c)
function vr_Uc(Ny,Nb,Vr,V0,Y,B,Price0,P,DC,C2,sumret,α)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
  
    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            if C[iy,ib,b] > 0
                c = C[iy,ib,b]
                C2[iy,ib,b] = CUDA.pow(c,(1-α)) / (1-α) + B[ib] - Price0[iy,b]*B[b] 
                #Note CUDA.pow only support certain types 
                #need to cast constant to Float32 instead of Float64
            end
    	end
	end
end

@benchmark @cuda threads=threadcount blocks=blockcount vr_Uc(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
```
One point worth noting is Julia CUDA's limitation in precision point. For example, CUDA.pow() only supports a few types of numerical inputs. Float64 constants need to be cast into Float32 before execution.



Since the ' operator is not supported in CUDA kernel, the calculation of sum of return requires one additional for loop, resulting in four for loops. Two for loops are reduced by the two-dimensional thread assignment, and two loops are contained in kernel calculations. This reduces the complexity from $O(n^4)$ to roughly $O(n^2)$. The straightforward two for-loop design will be shown to yield satisfying result in Benchmarking.

```julia
function vr_sumret(Ny,Nb,V0,P,sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            sumret[iy,ib,b] = 0
            for y in 1:Ny
                sumret[iy,ib,b] += P[iy,b]*V0[y,b]
            end
        end
    end

end
@benchmark @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
```

#Collecting the results, with loop fusion

### 6. Benchmarking

Julia provides the BenchmarkTools library, which attempts to provide straightforward solution to the error-prone and difficult task of benchmarking. In addition, benchmark Tools provide storage allocation to examine memory usage of the processes. While convenient and quick, BenchmarkTools library suffer from inaccuracy when the grid is granulized to a certain degree. We calculate the median time of multiple trials as the replacement in this case.

#### 6.1 Julia CUDA

We benchmarked each individual kernel's performance and the overall performance of the three steps in calculating Value of Default, Value of Repayment and Decision. The grid size for Endowment(Ny) * Bond(Nb) ranges from 50 * 50 to 500 * 500.

##### Value of Default

Value of Default requires a series of axpy operations, and we do them in the Julia way through the loop fusion operations. Loop fusion operation provides very intuitive and quick implementation of axpy operation, with speed on par with cublas' axpy operator. The calculation of temporary value matrix below showcase some possible operations of loop fusion, with much simpler syntax than cublas' axpy:

```julia
temp = P #assignment
temp .*= CUDA.transpose(A) #multiply in-place
temp .*= β #vector multiply scalar
```
##### Value of Repayment

Due to memory storage limits, 

#### Julia CUDA vs Julia



#### Julia CUDA vs C++ CUDA



[1] Unleash Julia
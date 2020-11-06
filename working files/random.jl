using FFTW
using BenchmarkTools
using CuArrays
CuArrays.allowscalar(true)

N = 100000
fftsz = 1024

# list of random vectors
A1_d = [CuArray(rand(ComplexF32, fftsz)) for i in 1:N]

# random 2D array
A2_d = CuArray(rand(ComplexF32, N, fftsz))

p1 = plan_fft!(A1_d[1])
p2 = plan_fft!(A2_d, 1)

function onefft!(data, plan)
    plan * data
end

#1
@btime CuArrays.@sync map(x -> onefft!(x, p1), $A1_d)
#2
@btime CuArrays.@sync mapslices(x -> onefft!(x, p1), $A2_d, dims=2)
#3
@btime CuArrays.@sync onefft!($A2_d, p2)

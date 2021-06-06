temp = P
temp .*= CUDA.transpose(A)
temp .*= beta

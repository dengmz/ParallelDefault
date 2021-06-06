x = (blockIdx().x-1)*blockDim().x + threadIdx().x
y = (blockIdx().y-1)*blockDim().y + threadIdx().y

function example(M)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if ( x <= Nx && y <= Ny)
        M[x,y] += 1
    end

    return # a return statement is necessary at the end of a kernel
end

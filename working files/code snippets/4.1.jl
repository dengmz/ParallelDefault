sum_default = CUDA.zeros(Ny)
# Ny is the size of grid points of possible values of y
for y in 1:Ny
    for iy in 1:Ny
        sumdefault[y] += f(y)*P[y,iy]
    end
end

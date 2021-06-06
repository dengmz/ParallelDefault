temp_vd = CUDA.zeros(Ny,Ny)
#Initialize
for y in 1:Ny
    for iy in 1:Ny
        temp_vd[y,iy] = f(y))*P[y,iy]
    end
end
sum_default = reduce(+, temp_vd, dims=2)

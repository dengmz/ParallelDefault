sumdef = CUDA.zeros(Ny)

A = ϕ* V0[:,1] + (1-ϕ)* Vd0
matrix = β* P .* CUDA.transpose(A)


sumdef = CUDA.zeros(Ny)
@cuda threads=50 def_init(sumdef,τ,Y,α)

temp = CUDA.zeros(Ny,Ny)

blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)

sumdef += reduce(+, temp, dims=2)

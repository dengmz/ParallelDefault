#Calculate Value of Default
function def_init(sumdef,tau,Y,alpha)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = CUDA.pow(exp((1-tau)*Y[i]),(1-alpha))/(1-alpha)
    end
    return
end

#adding expected value to sumdef
function def_add(matrix, P, beta, V0, Vd0, phi, Ny)
    y = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (iy <= Ny && y <= Ny)
        matrix[iy,y] = beta* P[iy,y]* (phi* V0[y,1] + (1-phi)* Vd0[y])
    end
    return
end

#finish calculation of Value of Default
function sumdef1(sumdef,Vd,Vd0,V0,phi,beta,P)
    A = phi* V0[:,1]
    A += (1-phi)* Vd0
    A.= phi.* V0[:,1] .+ (1-phi).* Vd0
    temp = P
    temp .*= CUDA.transpose(A)
    temp .*= beta
    sumdef += reduce(+, temp, dims=2) #This gives Vd
    Vd = sumdef
end

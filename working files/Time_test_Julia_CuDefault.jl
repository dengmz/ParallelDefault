#Method 1 of Vd
function sumdef1(sumdef)
    #sumdef = CUDA.zeros(Ny)
    A = ϕ* V0[:,1] + (1-ϕ)* Vd0
    temp = β* P .* CUDA.transpose(A)
    sumdef += reduce(+, temp, dims=2) #This gives Vd
    #Then do a value transport to Vd
end

#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = CUDA.pow(CUDA.exp((1-τ)*Y[i]),(1-α))/(1-α)
    end
    return
end

#adding expected value to sumdef
function def_add(matrix, P, β, V0, Vd0, ϕ, Ny)
    y = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (iy <= Ny && y <= Ny)
        matrix[iy,y] = β* P[iy,y]* (ϕ* V0[y,1] + (1-ϕ)* Vd0[y])
        #Note memory transfer of matrices of P and Vd0 are not optimal
    end
    return
end

#Method 2 of Vd
function sumdef2(sumdef) #Calculate sumdef in a kernel
    @cuda threads=threadcount blocks=blockcount def_init(sumdef,τ,Y,α)
    temp = CUDA.zeros(Ny,Ny)
    blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
    @cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)
    sumdef += reduce(+, temp, dims=2)
end

@benchmark sumdef1(sumdef) #240.6 μs
@benchmark sumdef2(sumdef)
#----

#Calculate Cost Matrix C
#19.3 μs
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

#map C -> U(C), then add β*sumret
#21.299 μs
function vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            if C[iy,ib,b] > 0
                c = C[iy,ib,b]
                C2[iy,ib,b] = CUDA.pow(c,(1-α)) / (1-α) + B[ib] - Price0[iy,b]*B[b] #Note CUDA.pow only support certain types, need to cast constant to Float32 instead of Float64
            end
        end
    end
end
@benchmark @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)

#Or instead of vr_C2 we can broadcast, but this is using scalar operation
#31.98 s
@benchmark C2 = U2.(C)

#----
#Calcuate sumret[iy,ib,b]
#This is a four-loop here, works, but not elegant, anyway to use the ' operator as in the CPU version?
#18.101 μs
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

#vr = U(c) + β * sumret
#45.3 μs
@benchmark vr = C2 + β * sumret

#Get max for [iy,ib,:]
#
@benchmark reduce(max,vr,dims=3)

#---
#write into decision function
#12.301 μs
function decide(Ny,Nb,Vd,Vr,V,decision)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        if (Vd[iy] < Vr[iy,ib])
            V[iy,ib] = Vr[iy,ib]
            decision[iy,ib] = 0
        else
            V[iy,ib] = Vd[iy]
            decision[iy,ib] = 1
        end
    end
    return
end
@benchmark @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)

#12 μs
function prob_calc(Ny,Nb,prob,P,decision)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        #prob[iy,ib] = P[iy,:]'decision[:,ib]
        for y in Ny
            prob[iy,ib] += P[iy,y]*decision[y,ib]
        end
    end
    return
end
@benchmark @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)

#15.4 μs
Price_calc(x, rstar) = (1-x) / (1+rstar)
@benchmark Price = Price_calc.(prob, rstar)

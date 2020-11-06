#Method 1 of Vd
sumdef = CUDA.zeros(Ny)
A = ϕ* V0[:,1] + (1-ϕ)* Vd0
matrix = β* P .* CUDA.transpose(A)
sumdef += reduce(+, matrix, dims=2)

#Method 2 of Vd
sumdef = CUDA.zeros(Ny)
@cuda threads=50 def_init(sumdef,τ,Y,α)

temp = CUDA.zeros(Ny,Ny)

blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)

sumdef += reduce(+, temp, dims=2)


function vr_cuda_pt_1(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        #temp = CUDA.zeros(1)
        for b in 1:Nb
            C[iy,ib,b] = -Price0[iy,b]*B[b] + CUDA.exp(Y[iy]) + B[ib]
            #C[iy,ib,:] = -CUBLAS.dot(Price0[iy,:],B) .+ CUDA.exp(Y[iy]) .+ B[ib]
            #element-wise multiplication, element-wise axpy
        end
        #sumret[iy,ib,:] = transpose(P[iy,:]'V0)
    end
end
#CUBLAS.cublasGemmEx

CUBLAS.dot(P[iy,:],V0)
@cuda threads=threadcount blocks=blockcount vr_cuda_pt_1(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)

function vr_C(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            C[iy,ib,b] = -Price0[iy,b]*B[b] + CUDA.exp(Y[iy]) + B[ib]
        end
    end
end
@cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)

function vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            #c= C[iy,ib,b]
            if c >= 0
                C2[iy,ib,b] = 1#CUDA.pow(c,(1-α)) / (1-α)
                #vr = CUDA.pow(c,(1-α))/(1-α) + β * sumret
            end
        end
    end
end
@cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)


C2 = U2.(C)

function vr_sumret(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Ny
            sumret[iy,ib,b] = P[iy,b]*V0[y,b]
        end
    end
end
@cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)



function vr_VR(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret,VR,β,U2)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            VR[iy,ib,b] = U2(C[iy,ib,b]) + β * sumret[iy,ib,b]
        end
    end
end

@cuda threads=threadcount blocks=blockcount vr_VR(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret,VR,β,U2)


function vr_Max(Ny,Nb,Vr,V0,Y,B,Price0,P,C,sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb

        end
    end
end



function vr_cuda_pt_2(Nb,Ny,Vr,V0,Y,B,Price0,P,C,sumret)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        Max = -999 #temp treatment
        for b in 1:Nb
            VR = map(U,C[iy,ib,:]) + β * sumret[iy,ib,:]
            #mapping, axpy
            vr = maximum(VR)
            Max = CUDA.max(Max, vr)
        end
        Vr[iy,ib] = Max
    end
    return
end

@btime C2 = U2.(C)
#Or we can do
@btime VR = C2 + β * sumret

blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount vr_cuda_pt_2(Nb,Ny,Vr,V0,Y,B,Price0,P,C,sumret)


function Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

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

        prob[iy,ib] = P[iy,:]'decision[:,ib]
        Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)
    end
    return
end

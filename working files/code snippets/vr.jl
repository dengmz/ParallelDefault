#Calculate Cost Matrix C
function vr_C(Ny,Nb,Y,B,Price0,P,C)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            C[iy,ib,b] = -Price0[iy,b]*B[b] + CUDA.exp(Y[iy]) + B[ib]
        end
    end
end

#map C -> U(C), then add β*sumret
function vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,alpha)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            if C[iy,ib,b] > 0
                c = C[iy,ib,b]
                C2[iy,ib,b] = CUDA.pow(c,(1-alpha)) / (1-alpha) + B[ib] - Price0[iy,b]*B[b] #Note CUDA.pow only support certain types, need to cast constant to Float32 instead of Float64
            end
        end
    end
end

#Calcuate sumret[iy,ib,b]
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

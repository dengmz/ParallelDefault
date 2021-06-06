#Calculate decision
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

#Calculate probability matrix
function prob_calc(Ny,Nb,prob,P,decision)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for y in Ny
            prob[iy,ib] += P[iy,y]*decision[y,ib]
        end
    end
    return
end

for b in 1:Nb
    c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
    if c > 0
        for y in 1:Ny
            sumret += P[iy,y]*V0[y,b]
        end
        vr = U(c) + beta * sumret
        Max = max(Max, vr)
    end
end

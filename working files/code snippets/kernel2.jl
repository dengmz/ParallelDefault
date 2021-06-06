function sumdef1(sumdef,Vd,Vd0,V0,ϕ,β,P)
    A = ϕ* V0[:,1]
    A += (1-ϕ)* Vd0
    A.= ϕ.* V0[:,1] .+ (1-ϕ).* Vd0
    temp = P
    temp .*= CUDA.transpose(A)
    temp .*= β
    sumdef += reduce(+, temp, dims=2) #This gives Vd
    Vd = sumdef
end

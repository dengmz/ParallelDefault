A .= phi* V0[:,1]
A .+= (1-phi)* Vd0
A.= phi.* V0[:,1] .+ (1-phi).* Vd0
temp = P
temp .*= CUDA.transpose(A)
temp .*= beta

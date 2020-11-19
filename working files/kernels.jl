using Random, Distributions
using CUDA
using BenchmarkTools, Base.Threads
#Make a jupyter notebook of time test when back home
#Method 1 of Vd
#A += (1-ϕ)* Vd0
function sumdef1(sumdef,Vd,Vd0,V0,ϕ,β,P)
    #sumdef = CUDA.zeros(Ny)
    A = ϕ* V0[:,1]
    A += (1-ϕ)* Vd0
    A.= ϕ.* V0[:,1] .+ (1-ϕ).* Vd0
    temp = P
    temp .*= CUDA.transpose(A)
    temp .*= β
    #temp = β* P .* CUDA.transpose(A)
    sumdef += reduce(+, temp, dims=2) #This gives Vd
    #Then do a value transport to Vd
    Vd = sumdef
end

#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = CUDA.pow(exp((1-τ)*Y[i]),(1-α))/(1-α)
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

#@benchmark sumdef1(sumdef) #240.6 μs
#@benchmark sumdef2(sumdef)
#----

#Calculate Cost Matrix C
#19.3 μs 100*80
#88.2 μs 200*150
function vr_C(Ny,Nb,Y,B,Price0,P,C)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            C[iy,ib,b] = -Price0[iy,b]*B[b] + CUDA.exp(Y[iy]) + B[ib]
        end
    end
end
#@benchmark @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)

#map C -> U(C), then add β*sumret
#21.299 μs 100*80
#113.3 μs 200*150
#test vr_C2 speed if
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
#@benchmark @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)

#Or instead of vr_C2 we can broadcast, but this is using scalar operation
#31.98 s 100*80
#296.114 ms
#@benchmark C2 = U2.(C)
#360.28 ms
#@benchmark C2 = U.(C)
#=
function vr_add(C2, β, sumret)
    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        for b in 1:Nb
            vr[iy,ib,b] = C2[iy,ib,ib] + β*sumret[iy,ib,ib]
        end
    end
end

@time @cuda threads=threadcount blocks=blockcount vr_add(C2, β, sumret)
vr = C2 + β * sumret
=#
#----
#Calcuate sumret[iy,ib,b]
#This is a four-loop here, works, but not elegant, anyway to use the ' operator as in the CPU version?
#18.101 μs 100*80
#85.4 μs 200*150
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
#@benchmark @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)

#vr = U(c) + β * sumret
#saxpy operation
#45.3 μs 100*80
#Slow μs 200*150
#@benchmark vr = C2 + β * sumret

#Get max for [iy,ib,:]
#
#@benchmark reduce(max,vr,dims=3)

#provide a benchmark here between reduction and kernel

#=
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
=#

#---
#write into decision function
#12.301 μs 100*80
#15.3 μs 200*150
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
#@benchmark @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)

#12 μs 100*80
#15.4 μs 200*150
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
#@benchmark @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)

#15.4 μs 100*80
#15.6 μs 200*150
Price_calc(x, rstar) = (1-x) / (1+rstar)
#@benchmark Price = Price_calc.(prob, rstar)


#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy] #BATCH UPDATE
function def_init_old(sumdef,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = exp((1-τ)*Y[i])/(1-α)
    end
    return
end

#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add_old(matrix, P, β, V0, Vd0, ϕ, Ny)
    y = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    #@cuprintln("iy=$iy,y=$y,stride1=$stride1,stride2=$stride2")
    #Create 1 matrix and substract when an indice is calcualted, check if remaining matrix is
    #@cuprintln("iy=$iy, y=$y")

    if (iy <= Ny && y <= Ny)
        matrix[iy,y] = β* P[iy,y]* (ϕ* V0[y,1] + (1-ϕ)* Vd0[y])
    end
    return
end

#line 8 Calculate Vr, still a double loop inside, tried to flatten out another loop
#Is it Markov Chain
#=
function vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        Max = -Inf
        for b in 1:Nb
            c = CUDA.exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b] #C[iy,ib,b]
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for y in 1:Ny
                    sumret += V0[y,b]*P[iy,y] #sumret[iy,ib,b] = Sum y=1 to Ny V0[y,b] * P[iy,y]
                end

                #vr = CUDA.pow(c,(1-α))/(1-α) + β * sumret
                vr = c^(1-α)/(1-α) + β * sumret
                Max = CUDA.max(Max, vr)
            end
        end
        Vr[iy,ib] = Max
    end
    return
end
=#

function vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        Max = -Inf
        for b in 1:Nb
            c = Float32(CUDA.exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b])
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for y in 1:Ny
                    sumret += V0[y,b]*P[iy,y]
                end
                Max = CUDA.max(Max, CUDA.pow(c,(1-α))/(1-α) + β * sumret)
            end
        end
        Vr[iy,ib] = Max
    end
    return
end


#line 9-14 debt price update
function Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

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

        for y in 1:Ny
            prob[iy,ib] += P[iy,y] * decision[y,ib]
        end

        Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)

    end
    return
end

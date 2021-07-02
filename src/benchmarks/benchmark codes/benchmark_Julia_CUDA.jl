#Modify the following parameters
Ny = 7 #number of endowment grid points
Nb = 1000 #number of bond grid points
sec = 20 #benchmark time limit for Value-of-Repayment
test_rounds = 10 #number of iterations inside the function for benchmarking

using Random, Distributions
using CUDA
using Base.Threads
using BenchmarkTools
#Initialization

#----Initialize Kernels
#line 7.1 Intitializing U((1-τ)iy) to each Vd[iy]
function def_init(sumdef,τ,Y,α)
    iy = threadIdx().x
    stride = blockDim().x
    for i = iy:stride:length(sumdef)
        sumdef[i] = CUDA.pow(exp((1-τ)*Y[i]),(1-α))/(1-α)
    end
    return
end

#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add(matrix, P, β, V0, Vd0, ϕ, Ny)
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
function vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)

        Max = -Inf
        for b in 1:Nb
            c = CUDA.exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
            if c > 0 #If consumption positive, calculate value of return
                sumret = 0
                for y in 1:Ny
                    sumret += V0[y,b]*P[iy,y]
                end

                vr = CUDA.pow(c,(1-α))/(1-α) + β * sumret
                Max = CUDA.max(Max, vr)
            end
        end
        Vr[iy,ib] = Max
    end
    return
end


#line 9-14 debt price update
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

        for y in 1:Ny
            prob[iy,ib] += P[iy,y] * decision[y,ib]
        end

        Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)

    end
    return
end


#Saxpy
function saxpy(X,Y,δ,Nb,Ny)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        X[iy,ib] = δ* X[iy,ib] + (1-δ)* Y[iy,ib]
    end
    return
end


#tauchen method for creating conditional probability matrix
function tauchen(ρ, σ, Ny, P)
    #Create equally spaced pts to fill into Z
    σ_z = sqrt((σ^2)/(1-ρ^2))
    Step = 10*σ_z/(Ny-1)
    Z = -5*σ_z:Step:5*σ_z

    #Fill in entries of 1~ny, ny*(ny-1)~ny^2
    for z in 1:Ny
        P[z,1] = cdf(Normal(), (Z[1]-ρ*Z[z] + Step/2)/σ)
        P[z,Ny] = 1 - cdf(Normal(),(Z[Ny] - ρ*Z[z] - Step/2)/σ)
    end

    #Fill in the middle part
    for z in 1:Ny
        for iz in 2:(Ny-1)
            P[z,iz] = cdf(Normal(), (Z[iz]-ρ*Z[z]+Step/2)/σ) - cdf(Normal(), (Z[iz]-ρ*Z[z]-Step/2)/σ)
        end
    end
end

#Setting parameters


maxInd = Ny * Nb #total grid points
rstar = 0.017 #r* used in price calculation
α = 0.5 #α used in utility function

#lower bound and upper bound for bond initialization
lbd = -1
ubd = 0

#β,ϕ,τ used as in part 4 of original paper
β = 0.953
ϕ = 0.282
τ = 0.5

δ = 0.8 #weighting average of new and old matrixs

#ρ,σ For tauchen method
ρ = 0.9
σ = 0.025


#Initializing Bond matrix
#B = zeros(Nb)
#B = CuArray{Float32}(undef,Nb)
minB = lbd
maxB = ubd
step = (maxB-minB) / (Nb-1)
B = CuArray(minB:step:maxB) #Bond

#Intitializing Endowment matrix
#Y = zeros(Ny)
σ_z = sqrt((σ^2)/(1-ρ^2))
Step = 10*σ_z/(Ny-1)
Y = CuArray(-5*σ_z:Step:5*σ_z) #Endowment

Pcpu = zeros(Ny,Ny)  #Conditional probability matrix
V = CUDA.fill(1/((1-β)*(1-α)),Ny, Nb) #Value
Price = CUDA.fill(1/(1+rstar),Ny, Nb) #Debt price
Vr = CUDA.zeros(Ny, Nb) #Value of good standing
Vd = CUDA.zeros(Ny) #Value of default
decision = CUDA.ones(Ny,Nb) #Decision matrix


U(x) = x^(1-α) / (1-α) #Utility function

#Initialize Conditional Probability matrix
tauchen(ρ, σ, Ny, Pcpu)
P = CUDA.zeros(Ny,Ny)
#P = CUDA.CUarray(Pcpu)
copyto!(P,Pcpu) #Takes long time

time_vd = 0
time_vr = 0
time_decide = 0
time_update = 0
time_init = 0

V0 = CUDA.deepcopy(V)
Vd0 = CUDA.deepcopy(Vd)
Price0 = CUDA.deepcopy(Price)
prob = CUDA.zeros(Ny,Nb)
decision = CUDA.ones(Ny,Nb)
decision0 = CUDA.deepcopy(decision)
threadcount = (32,32)
blockcount = (ceil(Int,Ny/32),ceil(Int,Ny/32))


#----Test starts

#Matrix to store benchmark results
Times = zeros(4)

function GPU_VD()
    for i in 1:test_rounds
        sumdef = CUDA.zeros(Ny)
        @cuda threads=32 def_init(sumdef,τ,Y,α)

        temp = CUDA.zeros(Ny,Ny)

        @cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)

        temp = sum(temp,dims=2)
        sumdef = sumdef + temp
        for i in 1:length(Vd)
            Vd[i] = sumdef[i]
        end
    end
end

t_vd = @benchmark GPU_VD()
Times[1] = median(t_vd).time/1e9/test_rounds
println("VD Finished")


function GPU_Decide()
    for i in 1:test_rounds
        @cuda threads=threadcount blocks=blockcount Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
    end
end

t_decide = @benchmark GPU_Decide()
Times[3] = median(t_decide).time/1e9/test_rounds
println("Decide Finished")

function GPU_Update()
    for i in 1:test_rounds
        err = maximum(abs.(V-V0)) #These are the main time consuming parts
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))

        @cuda threads=threadcount blocks=blockcount saxpy(Vd,Vd0,δ,1,Ny)
        @cuda threads=threadcount blocks=blockcount saxpy(Price,Price0,δ,Nb,Ny)
        @cuda threads=threadcount blocks=blockcount saxpy(V,V0,δ,Nb,Ny)
    end
end

t_update = @benchmark GPU_Update()
Times[4] = median(t_update).time/1e9/test_rounds
println("Update Half Finished")
println("Nb=",Nb,"Ny=",Ny)
println(Times)


function GPU_VR()
    for i in 1:test_rounds
        @cuda threads=threadcount blocks=blockcount vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
    end
end

t_vr = @benchmark GPU_VR() seconds = sec
Times[2] = median(t_vr).time/1e9/test_rounds

println("VR Finished")
#println(dump(t_vr))
println("Update Fully Finished")
println("Nb=",Nb,", Ny=",Ny)
println(Times)

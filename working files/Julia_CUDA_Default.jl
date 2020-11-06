using Random, Distributions
using CUDA
using BenchmarkTools, Base.Threads
#Initialization

#-----
#First Intialze the kernels

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

#= May add later, improve copy speed
function TauchenCopy(P,Pcpu,Nb,Ny)

    ib = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (ib <= Nb && iy <= Ny)
        P[iy,ib] = Pcpu[iy,ib]
    end
    return
end
=#

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
#Is it Markov Chain
function vr_temp(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

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

                vr = CUDA.pow(c,(1-α))/(1-α) + β * sumret
                Max = CUDA.max(Max, vr)
            end
        end
        Vr[iy,ib] = Max
    end
    return
end

blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
@cuda threads=threadcount blocks=blockcount vr_temp(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)



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

#-----
#Main starts


function main()

    #Setting parameters
    Ny = 100 #grid number of endowment
    Nb = 100 #grid number of bond
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
    minB = lbd
    maxB = ubd
    step = (maxB-minB) / (Nb-1)
    B = CuArray(minB:step:maxB) #Bond

    #Intitializing Endowment matrix
    σ_z = sqrt((σ^2)/(1-ρ^2))
    Step = 10*σ_z/(Ny-1)
    Y = CuArray(-5*σ_z:Step:5*σ_z) #Endowment

    Pcpu = zeros(Ny,Ny)  #Conditional probability matrix
    V = CUDA.fill(1/((1-β)*(1-α)), Ny, Nb) #Value
    Price = CUDA.fill(1/(1+rstar), Ny, Nb) #Debt price
    Vr = CUDA.zeros(Ny, Nb) #Value of good standing
    Vd = CUDA.zeros(Ny) #Value of default
    decision = CUDA.ones(Ny, Nb) #Decision matrix
    C = CUDA.zeros(Ny,Nb,Nb)
    sumret = CUDA.zeros(Ny,Nb,Nb)

    U(x) = x^(1-α) / (1-α) #Utility function, change this into a function

    #Initialize Conditional Probability matrix
    tauchen(ρ, σ, Ny, Pcpu)
    P = CUDA.zeros(Ny,Ny)
    copyto!(P,Pcpu) ####Takes long time


    #=
    threadcount = (30,30)
    blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
    @cuda threads=threadcount blocks=blockcount TauchenCopy(P,Pcpu,Nb,Ny)
=#


    err = 2000 #error
    tol = 1e-3 #error toleration
    iter = 0
    maxIter = 500 #Maximum interation

#-------
#Based on Paper Part4, Sovereign meets C++

    #line 3
    while (err > tol) & (iter < maxIter)
        #Keeping copies of Value, Value of defualt, Price for the previous round
        V0 = CUDA.deepcopy(V)
        Vd0 = CUDA.deepcopy(Vd)
        Price0 = CUDA.deepcopy(Price)
        prob = CUDA.zeros(Ny,Nb)
        decision = CUDA.ones(Ny,Nb)
        decision0 = CUDA.deepcopy(decision)

        threadcount = (30,30) #set up defualt thread numbers per block

        #line 7
        sumdef = CUDA.zeros(Ny)
        @cuda threads=50 def_init(sumdef,τ,Y,α)

        temp = CUDA.zeros(Ny,Ny)

        blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
        @cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)

        #Added this part for speed, may not work so well and untidy
        temp = sum(temp,dims=2)
        sumdef = sumdef + temp
        for i in 1:length(Vd)
            Vd[i] = sumdef[i]
        end



        #line 8

        blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
        @cuda threads=threadcount blocks=blockcount vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

        #line 9-14

        blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
        @cuda threads=threadcount blocks=blockcount Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

        #line 16
        #update Error and value matrix at round end

        err = maximum(abs.(Array(V)-Array(V0)))
        PriceErr = maximum(abs.(Array(Price)-Array(Price0)))
        VdErr = maximum(abs.(Array(Vd)-Array(Vd0)))

####Matrix Opertion May take too long time
        Vd = δ * Vd + (1-δ) * Vd0
        Price = δ * Price + (1-δ) * Price0
        V = δ * V + (1-δ) * V0

        iter += 1
        println("Errors of round $iter: Value error: $err, price error: $PriceErr, Vd error: $VdErr")

    end

    #Print final results
    println("Total Round ",iter)

    Vd = Vd[:,:]

    println("Vr: ====================")
    display(Vr)
    println("Vd: ==================")
    display(Vd)
    println("Decision: ==================")
    display(decision)
    println("Price: ==================")
    display(Price)

    return Vr,Vd,decision,Price

end

#Store Value of good standing, Value of default, Decision and Price matrix
@benchmark VReturn, VDefault, Decision, Price = main()

#-----
#Storing values as CSV
#=

using Parsers
using DataFrames
using CSV

dfPrice = DataFrame(Array(Price))
dfVr = DataFrame(Array(VReturn))
dfVd = DataFrame(Array(VDefault))
dfDecision = DataFrame(Array(Decision))

CSV.write("./Price.csv", dfPrice)
CSV.write("./Vr.csv", dfVr)
CSV.write("./Vd.csv", dfVd)
CSV.write("./Decision.csv", dfDecision)

using Random, Distributions
using CUDA
using Base.Threads
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


#-----
#Main starts


function main()

    #Setting parameters
    Ny = 50 #grid number of endowment
    Nb = 50 #grid number of bond
    maxIter = 2#Maximum interation
    in_iter = 3

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
    #=
    threadcount = (30,30)
    blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
    @cuda threads=threadcount blocks=blockcount TauchenCopy(P,Pcpu,Nb,Ny)
=#
    time_vd = 0
    time_vr = 0
    time_decide = 0
    time_update = 0
    time_init = 0

    err = 2000 #error
    tol = 1e-6 #error toleration
    iter = 0

#------
#Based on Paper Part4, Sovereign meets C++

    #line 3

        #Keeping copies of Value, Value of defualt, Price for the previous round
    while (err > tol) & (iter < maxIter)
    t = @timed begin
        V0 = CUDA.deepcopy(V)
        Vd0 = CUDA.deepcopy(Vd)
        Price0 = CUDA.deepcopy(Price)
        prob = CUDA.zeros(Ny,Nb)
        decision = CUDA.ones(Ny,Nb)
        decision0 = CUDA.deepcopy(decision)
        threadcount = (32,32) #set up defualt thread numbers per block
    end
    time_init += t[2]
        #line 7
#=
t = @timed begin
for i in 1:in_iter
    #=
        sumdef = CUDA.zeros(Ny)
        @cuda threads=32 def_init(sumdef,τ,Y,α)

        temp = CUDA.zeros(Ny,Ny)

        blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
        @cuda threads=threadcount blocks=blockcount def_add(temp, P, β, V0, Vd0, ϕ, Ny)
        #Added this part for speed, may not work so well and untidy
        temp = sum(temp,dims=2)
        sumdef = sumdef + temp
        for i in 1:length(Vd)
            Vd[i] = sumdef[i]
        end
    end=#

end
time_vd += t[2]

        #line 8
t = @timed begin
    #=
    for i in 1:in_iter
        blockcount = (ceil(Int,Nb/10),ceil(Int,Ny/10))
        @cuda threads=threadcount blocks=blockcount vr(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
    end
    =#
end
time_vr += t[2]
        #line 9-14
=#

#Now transform CUDA arrays into normal Arrays
Vd = Array(Vd)
V = Array(V)
decision = Array(decision)
Vr = Array(Vr)
prob = Array(prob)
Price = Array(Price)

t = @timed begin
    for i in 1:in_iter
        for ib in 1:Nb
            for iy = 1:Ny
                #Choose repay or default
                if (Vd[iy] < Vr[iy,ib])
                    V[iy,ib] = Vr[iy,ib]
                    decision[iy,ib] = 0
                else
                    V[iy,ib] = Vd[iy]
                    decision[iy,ib] = 1
                end

                #calculate debt price

                for y in 1:Ny
                    prob[iy,ib] += P[iy,y] * decision[y,ib]
                end
                Price[iy,ib] = (1-prob[iy,ib]) / (1+rstar)
            end
        end
    end
end
time_decide += t[2]

#line 16
#update Error and value matrix at round end
#=
t = @timed begin
    for i in 1:in_iter
        err = maximum(abs.(V-V0))
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))
        Vd = δ * Vd + (1-δ) * Vd0
        Price = δ * Price + (1-δ) * Price0
        V = δ * V + (1-δ) * V0
    end
end
time_update += t[2]
=#
iter += 1
println("Errors of round $iter")#: Value error: $err, price error: $PriceErr, Vd error: $VdErr")

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

    println("Time VD=", time_vd/iter/in_iter)
    println("Time VR=", time_vr/iter/in_iter)
    println("Time Decide=", time_decide/iter/in_iter)
    println("Time Update=", time_update/iter/in_iter)
    println("Time init=", time_init/iter)
    return Vr,Vd,decision,Price
end


#Store Value of good standing, Value of default, Decision and Price matrix
@time VReturn, VDefault, Decision, Price = main()

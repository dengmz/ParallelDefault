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

function main()

    #Setting parameters
    Ny = 7 #grid number of endowment
    Nb = 10000 #grid number of bond
    maxInd = Ny * Nb #total grid points
    rstar = Float32(0.017) #r* used in price calculation
    α = Float32(0.5) #α used in utility function

    #lower bound and upper bound for bond initialization
    lbd = -1
    ubd = 0

    #β,ϕ,τ used as in part 4 of original paper
    β = Float32(0.953)
    ϕ = Float32(0.282)
    τ = Float32(0.5)

    δ = Float32(0.8) #weighting average of new and old matrixs

    #ρ,σ For tauchen method
    ρ = Float32(0.9)
    σ = Float32(0.025)


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
    VR = CUDA.zeros(Ny,Nb,Nb)
    sumret = CUDA.zeros(Ny,Nb,Nb)

    #U(x) = x^(1-α) / (1-α) #Utility function, change this into a function
    function U(x)
        if x >= 0
            return x^(1-α) / (1-α) #Utility function7
        end
        return 0
    end

    function U2(x)
        return (x>0) * (x+0im)^(1-α) / (1-α)
    end
    #U2.(C)

    #Initialize Conditional Probability matrix
    tauchen(ρ, σ, Ny, Pcpu)
    P = CUDA.zeros(Ny,Ny)
    copyto!(P,Pcpu) ####Takes long time

    threadcount = (16,16) #set up defualt thread numbers per block
    blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))

    err = 2000 #error
    tol = 1e-3 #error toleration
    iter = 0
    maxIter = 50
    threadcount = (16,16) #set up defualt thread numbers per block
    blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))

    while (err > 1e-3) & (iter < maxIter)
        V0 = CUDA.deepcopy(V)
        Vd0 = CUDA.deepcopy(Vd)
        Price0 = CUDA.deepcopy(Price)
        prob = CUDA.zeros(Ny,Nb)
        decision = CUDA.ones(Ny,Nb)
        decision0 = CUDA.deepcopy(decision)
        C = CUDA.zeros(Ny,Nb,Nb)
        #We set up C2, sumret and sumdef in device memory
        sumret = CUDA.zeros(Ny,Nb,Nb)
        sumdef = CUDA.zeros(Ny)
        C2 = CUDA.zeros(Ny,Nb,Nb)

        @cuda threads=threadcount blocks=blockcount def_init(sumdef,τ,Y,α)
        #sumdef1(sumdef,Vd)
        @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
        @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
        @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
        sumret .*= β; vr = sumret; vr += C2 # vr = C2 + β * sumret
        Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
        @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
        @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
        Price = Price_calc.(prob, rstar)
#=

=#
        err = maximum(abs.(V-V0))
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))
        Vd *= δ; Vd0 *= (1-δ); Vd += Vd0 #Vd = δ * Vd + (1-δ) * Vd0
        Price *= δ; Price0 *= (1-δ); Price += Price0 #Price = δ * Price + (1-δ) * Price0
        V *= δ; V0 *= (1-δ); V += V0 #V = δ * V + (1-δ) * V0

        iter += 1
        #println("Errors of round $iter")
        println("Errors of round $iter: error: $err, price error: $PriceErr, Vd error: $VdErr")
    end
end

main()

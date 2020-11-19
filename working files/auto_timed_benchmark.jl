using Random, Distributions
using CUDA
using BenchmarkTools, Base.Threads

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

function bench()

    Grid_space = [600]
    #[50 100 150 200 300 400]# 450 500 550 600]
    global BenchResultsMedian = zeros(10,length(Grid_space))
    #sumdef1
    #vr_C
    #vr_C2
    #vr_sumret
    #vr=
    #reduce
    #decide
    #prob_calc

    global iter=1

    for i in Grid_space
        println("round $iter")
        Ny = i
        Nb = i
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
        C = CUDA.zeros(Ny,Nb,Nb)
        VR = CUDA.zeros(Ny,Nb,Nb)
        sumret = CUDA.zeros(Ny,Nb,Nb)
        V0 = CUDA.deepcopy(V)
        Vd0 = CUDA.deepcopy(Vd)
        Price0 = CUDA.deepcopy(Price)
        prob = CUDA.zeros(Ny,Nb)
        decision = CUDA.ones(Ny,Nb)
        decision0 = CUDA.deepcopy(decision)
        sumdef = CUDA.zeros(Ny)
        C2 = CUDA.zeros(Ny,Nb,Nb)
        global vr

        tauchen(ρ, σ, Ny, Pcpu)
        #global P = CUDA.zeros(Ny,Ny)
        #copyto!(P,Pcpu) ####Takes long time
        P = CuArray(Pcpu)

        threadcount = (16,16) #set up defualt thread numbers per block
        blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))

        println("begin benchmark")
        global counter = 30
        elem = 1
        @cuda threads=threadcount blocks=blockcount def_init(sumdef,τ,Y,α)

        t=[]
        for i in 1:counter
            time = @timed sumdef1(sumdef,Vd,Vd0,V0,ϕ,β,P)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed sumret .*= β; vr = sumret; vr += C2
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1

        t=[]
        for i in 1:counter
            time = @timed Price = Price_calc.(prob, rstar)
            push!(t, time[2])
        end
        #display(t)
        BenchResultsMedian[elem,iter] = median(t)
        elem+=1
        println("iter $iter over")
        iter+=1
        display(BenchResultsMedian)

    end

end



bench()

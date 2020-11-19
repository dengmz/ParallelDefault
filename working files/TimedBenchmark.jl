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

    Grid_space = [50 100 150 200 300 400 450 500 550 600]
    BenchResultsMedian = zeros(9,length(Grid_space))
    BenchResultsMemory = zeros(9,length(Grid_space))
    #sumdef1
    #vr_C
    #vr_C2
    #vr_sumret
    #vr=
    #reduce
    #decide
    #prob_calc

    iter=1

    for i in Grid_space
        println("round $iter")
        global Ny = i
        global Nb = i
        global maxInd = Ny * Nb #total grid points
        global rstar = Float32(0.017) #r* used in price calculation
        global α = Float32(0.5) #α used in utility function

        #lower bound and upper bound for bond initialization
        global lbd = -1
        global ubd = 0

        #β,ϕ,τ used as in part 4 of original paper
        global β = Float32(0.953)
        global ϕ = Float32(0.282)
        global τ = Float32(0.5)

        global δ = Float32(0.8) #weighting average of new and old matrixs

        #ρ,σ For tauchen method
        global ρ = Float32(0.9)
        global σ = Float32(0.025)


        #Initializing Bond matrix
        global minB = lbd
        global maxB = ubd
        step = (maxB-minB) / (Nb-1)
        global B = CuArray(minB:step:maxB) #Bond

        #Intitializing Endowment matrix
        global σ_z = sqrt((σ^2)/(1-ρ^2))
        global Step = 10*σ_z/(Ny-1)
        global Y = CuArray(-5*σ_z:Step:5*σ_z) #Endowment

        global Pcpu = zeros(Ny,Ny)  #Conditional probability matrix
        global V = CUDA.fill(1/((1-β)*(1-α)), Ny, Nb) #Value
        global Price = CUDA.fill(1/(1+rstar), Ny, Nb) #Debt price
        global Vr = CUDA.zeros(Ny, Nb) #Value of good standing
        global Vd = CUDA.zeros(Ny) #Value of default
        global C = CUDA.zeros(Ny,Nb,Nb)
        global VR = CUDA.zeros(Ny,Nb,Nb)
        global sumret = CUDA.zeros(Ny,Nb,Nb)
        global V0 = CUDA.deepcopy(V)
        global Vd0 = CUDA.deepcopy(Vd)
        global Price0 = CUDA.deepcopy(Price)
        global prob = CUDA.zeros(Ny,Nb)
        global decision = CUDA.ones(Ny,Nb)
        global decision0 = CUDA.deepcopy(decision)
        global sumdef = CUDA.zeros(Ny)
        global C2 = CUDA.zeros(Ny,Nb,Nb)
        global vr

        tauchen(ρ, σ, Ny, Pcpu)
        global P = CUDA.zeros(Ny,Ny)
        copyto!(P,Pcpu) ####Takes long time

        global threadcount = (16,16) #set up defualt thread numbers per block
        global blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))

        println("begin benchmark")
        @cuda threads=threadcount blocks=blockcount def_init(sumdef,τ,Y,α)
        display(sumdef)
        sumdef1(sumdef,Vd,V0)
        t = @benchmark sumdef1(sumdef,Vd,V0)
        BenchResultsMedian[1,iter] = time(median(t))
        BenchResultsMemory[1,iter] = memory(median(t))
        t=0;

        t = @benchmark @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
        BenchResultsMedian[2,iter] = time(median(t))
        BenchResultsMemory[2,iter] = memory(median(t))
        @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
        t=0;
        t = @benchmark @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
        BenchResultsMedian[3,iter] = time(median(t))
        BenchResultsMemory[3,iter] = memory(median(t))
        @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
        t=0;
        t = @benchmark @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
        BenchResultsMedian[4,iter] = time(median(t))
        BenchResultsMemory[4,iter] = memory(median(t))
        @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
        t=0;
        sumret0 = sumret;
        t = @benchmark sumret .*= β; vr = sumret; vr += C2
        BenchResultsMedian[5,iter] = time(median(t))
        BenchResultsMemory[5,iter] = memory(median(t))
        sumret = sumret0; sumret .*= β; vr = sumret; vr += C2
        t=0;
        t = @benchmark Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
        BenchResultsMedian[6,iter] = time(median(t))
        BenchResultsMemory[6,iter] = memory(median(t))
        Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
        t=0;
        t = @benchmark @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
        BenchResultsMedian[7,iter] = time(median(t))
        BenchResultsMemory[7,iter] = memory(median(t))
        @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
        t=0;
        t = @benchmark @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
        BenchResultsMedian[8,iter] = time(median(t))
        BenchResultsMemory[8,iter] = memory(median(t))
        @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
        t=0;
        t = @benchmark Price = Price_calc.(prob, rstar)
        BenchResultsMedian[9,iter] = time(median(t))
        BenchResultsMemory[9,iter] = memory(median(t))
        Price = Price_calc.(prob, rstar)
        t=0;
        iter+=1
        println("iter $iter over")
        display(BenchResultsMedian)
        display(BenchResultsMemory)

    end

end



bench()

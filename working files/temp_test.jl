#Benchmark on old implementation
#Using @benchmark
#add grid_space element N to test on N*N endowment*bond matrix
function bench_old_version()

    Grid_space = [250] #[50 100 150 200 300 400]# 450 500 550 600]
    global BenchResultsMedian = zeros(10,length(Grid_space))

    global iter=1

    for i in Grid_space
        println("round $iter")

        #Initialize varaibles
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
        elem = 1
        global temp = CUDA.zeros(Ny,Ny)
        tauchen(ρ, σ, Ny, Pcpu)
        P = CuArray(Pcpu)

        global threadcount = (16,16) #set up defualt thread numbers per block
        global blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))

        println("begin benchmark")


        #Part 1, get total time for value of default calculation, t0+t1+t2+t3
        t0 = @benchmark @cuda threads=50 def_init_old(sumdef,τ,Y,α)
        BenchResultsMedian[elem,iter] = time(median(t0))
        elem += 1

        t1 = @benchmark @cuda threads=threadcount blocks=blockcount def_add_old(temp, P, β, V0, Vd0, ϕ, Ny)
        BenchResultsMedian[elem,iter] = time(median(t1))
        elem += 1

        t2 = @benchmark temp2 = sum(temp,dims=2)
        global temp2 = sum(temp,dims=2)
        t3 = @benchmark sumdef2 = sumdef + temp2
        BenchResultsMedian[elem,iter] = time(median(t0)) + time(median(t1)) + time(median(t2)) + time(median(t3))
        elem += 1

        #Part 2, get total time for value of repayment calculation
        t = @benchmark @cuda threads=threadcount blocks=blockcount vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
        BenchResultsMedian[elem,iter] = time(median(t))
        elem += 1

        #Part 3, get total time for decision calculation
        t = @benchmark @cuda threads=threadcount blocks=blockcount Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
        BenchResultsMedian[elem,iter] = time(median(t))
        elem += 1

        println("iter $iter over")
        iter+=1
        display(BenchResultsMedian)

    end

end

bench_old_version()

using LoopVectorization
using BenchmarkTools
using Random, Distributions


#-----
#First a test on the best method to modify an array, avx loop, vmap and map

function def_init(sumdef,Y,α,τ)
    for i in 1:Ny
        sumdef[i] = exp((1-τ)*Y[i])^(1-α)/(1-α)
    end
end

function def_init_avx(sumdef,Y,α,τ)
    @avx for i in eachindex(Y)
        sumdef[i] = exp((1-τ)*Y[i])^(1-α)/(1-α)
    end
end

func_def_init(Y) = exp((1-τ)*Y)^(1-α)/(1-α)

#Testing different implementations, vmap&map the quickest
@benchmark def_init(sumdef,Y,α,τ)
@benchmark def_init_avx(sumdef,Y,α,τ)
@benchmark vmap!(func_def_init,$sumdef,$Y)
@benchmark map!(func_def_init,$sumdef,$Y)



#------
#Then second part of default value
#Quick enough, still much space for improvements
function def_add_norm(consdef,P,V0,Vd0)
    for iy in 1:Ny
        for y in 1:Ny
            consdef[iy] += β* P[iy,y]* (ϕ* V0[y,1] + (1-ϕ)* Vd0[y])
        end
    end
end

@benchmark def_add_norm(consdef,P,V0,Vd0)


#line 7.2 adding second expected part to calcualte Vd[iy]
function def_add_cuda(matrix, P, β, V0, Vd0, ϕ, Ny)
    y = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if (iy <= Ny && y <= Ny)
        matrix[iy,y] = β* P[iy,y]* (ϕ* V0[y,1] + (1-ϕ)* Vd0[y])
    end
    return
end

#----
#The heaviest chunck for calculation
#There are four loops, any way to divide the loops up
#Should calculation of Cost be separated out, but Cost may be too large for an array

function vr_norm(Vr,V0,Y,B,Price0,P)
    for i = 1:Nb*Ny
        ib = convert(Int64,ceil(i/Ny))
        iy = convert(Int64,i - (ib-1)*Ny)
        Max = -Inf

        for b in 1:Nb
            c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]

            if c > 0 # If consumption positive, calculate value of return
                # In one matrix operation fro line 77 to 87
                # How to find max, to turn to matrix, procedures
                # calculate individually, calculate by matrix
                # compare if have GPU, write this way, without GPU with multithreading
                sumret = P[iy,:]'V0[:,b]
                vr = U(c) + β * sumret
                Max = max(Max, vr)
            end
        end
        Vr[iy,ib] = Max
    end
end

vr_norm(Vr,V0,Y,B,Price0,P)




function vr_norm_C(Vr,V0,Y,B,Price0,P)
    for i = 1:Nb*Ny
        ib = convert(Int64,ceil(i/Ny))
        iy = convert(Int64,i - (ib-1)*Ny)
        Max = -Inf

        C = zeros(Ny,Nb,Nb)
        sumret = zeros(Ny,Nb,Nb)

        for b in 1:Nb
            C[iy,ib,:] = Price0[iy,:].*B

            #if c > 0 #How to do this line
                # If consumption positive, calculate value of return
                # In one matrix operation fro line 77 to 87
                # How to find max, to turn to matrix, procedures
                # calculate individually, calculate by matrix
                # compare if have GPU, write this way, without GPU with multithreading
                sumret[iy,ib,:] = transpose(P[iy,:]'V0)
                VR = map(U,C[iy,ib,:]) .+ β * sumret[iy,ib,:]
                vr = reduce(max,VR)
                Max = max(Max, vr)
            #end
        end
        Vr[iy,ib] = Max
    end
end

vr_norm_C(Vr,V0,Y,B,Price0,P)

#line 8 Calculate Vr, still a double loop inside, tried to flatten out another loop
function vr_cuda(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

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

#----
#Initial values set up


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


U(x) = x^(1-α) / (1-α) #Utility function


#Setting parameters
Ny = 50 #grid number of endowment
Nb = 50 #grid number of bond
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
B = minB:step:maxB #Bond

#Intitializing Endowment matrix
#Y = zeros(Ny)
σ_z = sqrt((σ^2)/(1-ρ^2))
Step = 10*σ_z/(Ny-1)
Y = -5*σ_z:Step:5*σ_z #Endowment
sumdef = zeros(Ny)
consdef = zeros(Ny)

Pcpu = zeros(Ny,Ny)  #Conditional probability matrix
V = fill(1/((1-β)*(1-α)),Ny, Nb) #Value
Price = fill(1/(1+rstar),Ny, Nb) #Debt price
Vr = zeros(Ny, Nb) #Value of good standing
Vd = zeros(Ny) #Value of default
decision = ones(Ny,Nb) #Decision matrix

V0 = deepcopy(V)
Vd0 = deepcopy(Vd)
Price0 = deepcopy(Price)
prob = zeros(Ny,Nb)
decision = ones(Ny,Nb)
decision0 = deepcopy(decision)

P = zeros(Ny,Ny)
tauchen(ρ, σ, Ny, P)

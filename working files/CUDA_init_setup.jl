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

#Setting parameters
Ny = 200 #grid number of endowment
Nb = 150 #grid number of bond
maxInd = Ny * Nb #total grid points
rstar = 0.017 #r* used in price calculation
α = Float32(0.5) #α used in utility function

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


err = 2000 #error
tol = 1e-3 #error toleration
iter = 0
maxIter = 500 #Maximum interation

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

threadcount = (16,16) #set up defualt thread numbers per block
blockcount = (ceil(Int,Ny/10),ceil(Int,Ny/10))
iy = 1
ib = 1

using LoopVectorization
using BenchmarkTools
using Random, Distributions

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


function U(x)
    if x >= 0
        return x^(1-α) / (1-α) #Utility function7
    end
    return 0
end

#Setting parameters
Ny = 7 #grid number of endowment
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
C = zeros(Ny,Nb,Nb)
sumret = zeros(Ny,Nb,Nb)

P = zeros(Ny,Ny)
tauchen(ρ, σ, Ny, P)

Ny = 7
Nb = 10000
C = zeros(Ny,Nb,Nb)
varinfo(r"C")

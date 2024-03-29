#Modify the following parameters
Ny = 7 #number of endowment grid points
Nb = 20000 #number of bond grid points
sec = 20 #benchmark time limit for Value-of-Repayment
test_rounds = 10 #number of iterations inside the function for benchmarking

using Random, Distributions
using BenchmarkTools
#Initialization

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


rstar = 0.017
lbd = -1
ubd = 0
rrisk = 0.5
β = 0.953
τ = 0.15
θ = 0.282
tol = 1e-10
ϕ = rrisk
δ = 0.8
ρ = 0.9
σ = 0.025
τ = 0.5


B = zeros(Nb)
Y = zeros(Ny)
σ_z = sqrt((σ^2)/(1-ρ^2))
Step = 10*σ_z/(Ny-1)
Y = -5*σ_z:Step:5*σ_z


P = zeros(Ny,Ny)
V = fill(1/((1-β)*(1-rrisk)),Ny, Nb)
Price = fill(1/(1+rstar),Ny, Nb)
Vr = zeros(Ny, Nb)
Vd = zeros(Ny)
decision = ones(Ny,Nb)

U(x) = x^(1-ϕ) / (1-ϕ)

#Initialize Bond grid
minB = lbd
maxB = ubd
step = (maxB-minB) / (Nb-1)
B = minB:step:maxB

#Initialize Shock grid
tauchen(ρ, σ, Ny, P)
sumdef = 0

err = 2000
tol = 1e-6
iter = 0

time_vd = 0
time_vr = 0
time_decide = 0

V0 = deepcopy(V)
Vd0 = deepcopy(Vd)
Price0 = deepcopy(Price)
prob = zeros(Ny,Nb)

#3

#-------Test starts
Times = zeros(4)

#=5
function CPU_VD()
    for i in 1:test_rounds
        for iy = 1:Ny
            sumdef = U(exp((1-τ)*Y[iy]))
            for y in 1:Ny
                sumdef += (β* P[iy,y]* (θ* V0[y,1] + (1-θ)* Vd0[y]))
            end
            Vd[iy] = sumdef
        end
    end
end

t_vd = @benchmark CPU_VD()
Times[1] = median(t_vd).time/1e9/test_rounds
println("VD Finished")
=#
function CPU_Update(V,V0,Price,Price0,Vd,Vd0,δ)
    for i in 1:test_rounds
        err = maximum(abs.(V-V0))
        PriceErr = maximum(abs.(Price-Price0))
        VdErr = maximum(abs.(Vd-Vd0))
        Vd = δ * Vd + (1-δ) * Vd0
        Price = δ * Price + (1-δ) * Price0
        V = δ * V + (1-δ) * V0
    end
end
t_update = @benchmark CPU_Update(V,V0,Price,Price0,Vd,Vd0,δ)
Times[4] = median(t_update).time/1e9/test_rounds
println("Update Finished")

function CPU_Decide()
    for i in 1:test_rounds
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

t_decide = @benchmark CPU_Decide()
Times[3] = median(t_decide).time/1e9/test_rounds
println("Decide Finished")

println("Update Half Finished")
println("Nb=",Nb,"Ny=",Ny)
println(Times)
#=
function CPU_VR()
    for i in 1:test_rounds
        for ib in 1:Nb
            for iy = 1:Ny
                Max = -Inf
                for b in 1:Nb
                    c = exp(Y[iy]) + B[ib] - Price0[iy,b]*B[b]
                    if c > 0
                        sumret = 0
                        for y in 1:Ny
                            sumret += P[iy,y]*V0[y,b]
                        end
                        vr = U(c) + β * sumret
                        Max = max(Max, vr)
                    end
                end
            Vr[iy,ib] = Max
            end
        end
    end
end

t_vr = @benchmark CPU_VR() seconds = sec
Times[2] = median(t_vr).time/1e9/test_rounds
println("VR Finished")

#println(dump(t_vr))
println("Update Fully Finished")
println("Nb=",Nb,", Ny=",Ny)
println(Times)
=#

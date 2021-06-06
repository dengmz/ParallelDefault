using Random, Distributions
using CUDA
using BenchmarkTools, Base.Threads

#Default
d0() = @cuda threads=50 def_init(sumdef,τ,Y,α)
d = @benchmark d0() seconds=10 gcsample=true

sd() = sumdef1(sumdef,Vd,Vd0,V0,ϕ,β,P)
d1 = @benchmark sd() seconds=10 gcsample=true

#Replacement
vrc() = @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
r1 = @benchmark vrc() seconds=10 gcsample=true

vrc2() = @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
r2 = @benchmark vrc2() seconds=10 gcsample=true

vrs() = @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
r3 = @benchmark vrs() seconds=10 gcsample=true

r4 = @benchmark sumret .*= β seconds=10 gcsample=true
r5 = @benchmark vr = sumret seconds=10 gcsample=true
global vr = sumret
r6 = @benchmark vr += C2 seconds=10 gcsample=true

rd() = reduce(max,vr,dims=3)
r7 = @benchmark rd() seconds=10 gcsample=true

#Decide
dec() = @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
de1 = @benchmark dec() seconds=10 gcsample=true

prb() = @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
de2 = @benchmark prb() seconds=10 gcsample=true

pr = @benchmark Price = Price_calc.(prob, rstar) seconds=10 gcsample=true

defaddold() =  @benchmark @cuda threads=threadcount blocks=blockcount def_add_old(temp, P, β, V0, Vd0, ϕ, Ny)
daold0 = @benchmark defaddold() seconds=20 gcsample=true

vrold() = @cuda threads=threadcount blocks=blockcount vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
vrold0 = @benchmark vrold() seconds=20 gcsample=true

decideold() = @cuda threads=threadcount blocks=blockcount Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
deold0 = @benchmark decideold() seconds=20 gcsample=true

save("./NEW100.jld","daold0",daold0,"vrold0",vrold0,"deold0",deold0)

oldvar500 = load("./NEW100.jld")





#=
using Parsers
using DataFrames
using CSV
=#
using JLD
using BenchmarkTools
save("./var500.jld","d1",d1,"r1",r1,"r2",r2,"r3",r3, "r4",r4,"r7",r7,"de1",de1,"de2",de2,"pr",pr)
var500 = load("./var500.jld")
d = load("./var.jld")
jldopen("var.jld", "w") do file
    write(file, "pr", pr)  # alternatively, say "@write file A"
end

d_old() = @cuda threads=threadcount blocks=blockcount def_add_old(temp, P, β, V0, Vd0, ϕ, Ny)
do1 = @benchmark d_old() seconds=10 gcsample=true
d_old_2() = sum(temp,dims=2)
do2 = @benchmark d_old2() seconds=10 gcsample=true
r_old() = @cuda threads=threadcount blocks=blockcount vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
ro1 = @benchmark r_old() seconds=10 gcsample=true
dec_old() = @cuda threads=threadcount blocks=blockcount Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
deco1 = @benchmark dec_old() seconds=10 gcsample=true

d1 = var500["d1"]
r1 = var500["r1"]
r2 = var500["r2"]
r3 = var500["r3"]
r4 = var500["r4"]
r7 = var500["r7"]
de1 = var500["de1"]
de2 = var500["de2"]
pr = var500["pr"]

r1t = r1.times
r = r1.times[1:13] + r2.times[1:13] + r3.times[1:13] + r4.times[1:13] + r7.times[1:13]
de = de1.times + de2.times

using PyPlot
const plt = PyPlot
boxplot(r1t)

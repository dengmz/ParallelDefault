using Random, Distributions
using CUDA
using BenchmarkTools, Base.Threads


# μs 300*300
# μs 500*500
#Set up Benchmark Parameters
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

@benchmark @cuda threads=threadcount blocks=blockcount def_init(sumdef,τ,Y,α)


#230.4 μs 50*50
#268 μs 200*150
#608 μs 300*300
#761 μs 500*500
@benchmark sumdef1(sumdef,Vd) seconds=5 #240.6 μs
i = @time sumdef1(sumdef,Vd)

@benchmark sumdef2(sumdef)

#Calculate Cost Matrix C
#19.1 μs 50*50
#19.3 μs 100*100
#88.2 μs 200*200
#90.95 μs 300*300
#163 μs 500*500
@benchmark @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
@time @cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)

#map C -> U(C), then add β*sumret
#20.299 μs 50*50
#21.299 μs 100*80
#113.3 μs 200*150
#177.5 μs 300*300
#248 μs 500*500
#test vr_C2 speed if
@benchmark @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
@time @cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)

#Or instead of vr_C2 we can broadcast, but this is using scalar operation
#6.791 μs 50*50
#31.98 s 100*80
#296.114 ms 200*150
#X μs 300*300
#X μs 500*500
@benchmark C2 = U2.(C)
@time C2 = U2.(C)

#Calcuate sumret[iy,ib,b]
#This is a four-loop here, works, but not elegant, anyway to use the ' operator as in the CPU version?
#17.4 μs 50*50
#18.101 μs 100*80
#85.4 μs 200*150
#177.2 μs 300*300
#222.4 μs 500*500
@benchmark @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
@time @cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)

#vr = U(c) + β * sumret
#saxpy operation
#18.7 μs 50*50
#18.5 μs 100*80
#1 μs 200*150
#194 μs 300*300
#255/84300 μs 500*500
t = @benchmark sumret .*= β; vr = sumret; vr += C2
i = @time sumret .*= β; vr = sumret; vr += C2

#Get max for [iy,ib,:]
#168.4 μs 50*50
#360 μs 200*150
#480.9 μs 300*300
#785.3 μs 500*500
@benchmark reduce(max,vr,dims=3)
@time reduce(max,vr,dims=3)

#write into decision function
#34.1 μs 50*50
#12.301 μs 100*80
#167 μs 200*150
#187.2 μs 300*300
#224.7 μs 500*500
@benchmark @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
@time @cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)

#32.6 μs 50*50
#12 μs 100*80
#15.4 μs 200*150
#156 μs 300*300
#168 μs 500*500
@benchmark @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
@time @cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)

#38.4 μs 50*50
#189 μs 100*80
#210 μs 200*150
#620.8 μs 300*300
#864.1 μs 500*500
Price_calc(x, rstar) = (1-x) / (1+rstar)
@benchmark Price = Price_calc.(prob, rstar)
@time Price = Price_calc.(prob, rstar)

#999 μs 500*500
@time err = maximum(abs.(V-V0))
#1022 μs 500*500
@time PriceErr = maximum(abs.(Price-Price0))
#890 μs 500*500
@time VdErr = maximum(abs.(Vd-Vd0))
#346 μs 500*500
@time Vd *= δ; Vd0 *= (1-δ); Vd += Vd0 #Vd = δ * Vd + (1-δ) * Vd0
#249 μs 500*500
@time Price *= δ; Price0 *= (1-δ); Price += Price0 #Price = δ * Price + (1-δ) * Price0
#204 μs 500*500
@time V *= δ; V0 *= (1-δ); V += V0 #V = δ * V + (1-δ) * V0

#19.1 μs 50*50
#19.3 μs 100*100
#88.2 μs 200*200
#90.95 μs 300*300
#163 μs 500*500
@benchmark def_init_cpu(sumdef,Y,α,τ)
@time def_init_cpu(sumdef,Y,α,τ)

#19.1 μs 50*50
#19.3 μs 100*100
#88.2 μs 200*200
#90.95 μs 300*300
#163 μs 500*500
@benchmark def_add_norm_cpu(consdef,P,V0,Vd0)
@time def_add_norm_cpu(consdef,P,V0,Vd0)

#19.1 μs 50*50
#19.3 μs 100*100
#88.2 μs 200*200
#90.95 μs 300*300
#163 μs 500*500
@benchmark vr_norm_cpu(Vr,V0,Y,B,Price0,P)
@time vr_norm_cpu(Vr,V0,Y,B,Price0,P)

@benchmark Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
@time Decide(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

@benchmark @cuda threads=50 def_init_old(sumdef,τ,Y,α)
@time @cuda threads=50 def_init_old(sumdef,τ,Y,α)

t = @benchmark @cuda threads=threadcount blocks=blockcount def_add_old(temp, P, β, V0, Vd0, ϕ, Ny)
@time @cuda threads=threadcount blocks=blockcount def_add_old(temp, P, β, V0, Vd0, ϕ, Ny)
@benchmark temp2 = sum(temp,dims=2)
temp2 = sum(temp,dims=2)
@benchmark sumdef2 = sumdef + temp2

@benchmark for i in 1:length(Vd)
    Vd[i] = sumdef[i]
end

@benchmark Vd = sumdef

#19.1 μs 50*50
#19.3 μs 100*100
#88.2 μs 200*200
#90.95 μs 300*300
#163 μs 500*500
@benchmark @cuda threads=threadcount blocks=blockcount vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)
@time @cuda threads=threadcount blocks=blockcount vr_old(Nb,Ny,α,β,τ,Vr,V0,Y,B,Price0,P)

@benchmark @cuda threads=threadcount blocks=blockcount Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)
@time Decide_old(Nb,Ny,Vd,Vr,V,decision,decision0,prob,P,Price,rstar)

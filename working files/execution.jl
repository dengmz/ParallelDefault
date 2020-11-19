sumdef1(sumdef,Vd)
#sumdef2(sumdef)
@cuda threads=threadcount blocks=blockcount vr_C(Ny,Nb,Y,B,Price0,P,C)
@cuda threads=threadcount blocks=blockcount vr_C2(Ny,Nb,Vr,V0,Y,B,Price0,P,C,C2,sumret,α)
@cuda threads=threadcount blocks=blockcount vr_sumret(Ny,Nb,V0,P,sumret)
vr = C2 + β * sumret
Vr = reshape(reduce(max,vr,dims=3),(Ny,Nb))
@cuda threads=threadcount blocks=blockcount decide(Ny,Nb,Vd,Vr,V,decision)
@cuda threads=threadcount blocks=blockcount prob_calc(Ny,Nb,prob,P,decision)
Price = Price_calc.(prob, rstar)

#17.9 μs 100*80
@benchmark def_init(sumdef,Y,α,τ)
@time def_init(sumdef,Y,α,τ)
#2249 μs 100*80
@benchmark def_add_norm(consdef,P,V0,Vd0)
@time def_add_norm(consdef,P,V0,Vd0)
#1.517 s = 1517000 μs 100*80
@benchmark vr_norm(Vr,V0,Y,B,Price0,P)
@time vr_norm(Vr,V0,Y,B,Price0,P)
#7.339s = 7339000 μs 100*80 #reduce is costing a lot of computation power
@benchmark vr_norm_C(Vr,V0,Y,B,Price0,P)
@time vr_norm_C(Vr,V0,Y,B,Price0,P)

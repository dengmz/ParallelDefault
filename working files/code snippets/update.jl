err = maximum(abs.(V-V0))
PriceErr = maximum(abs.(Price-Price0))
VdErr = maximum(abs.(Vd-Vd0))
f(x,y) = delta * x + (1-delta) * y
Vd .= f.(Vd,Vd0)
Price .= f.(Price,Price0)
V .= f.(V,V0)

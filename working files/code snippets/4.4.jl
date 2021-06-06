temp = beta * P .* CUDA.transpose(phi.*V[y',0] .+
      (1-phi).*V_d[y',i_y])
Vd .= sumdef .+ reduce(+, temp, dims=2)

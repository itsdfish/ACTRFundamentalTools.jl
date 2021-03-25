using Parameters, Distributions, StatsFuns

function simulate(N; s, blc)
    model = construct_model(blc, s)
    return rand(model, N)
end

function construct_model(α, s)
  λ = s*π/sqrt(3)
  μ,σ = convolve_normal(motor=(μ=.21,N=1), cr=(μ=.05,N=11),visual=(μ=.085,N=2),
    imaginal=(μ=.2,N=1))
  model = Normal(μ, σ) + LogNormal(-α, λ) + LogNormal(-α, λ)
  return model
end

function loglike(blc, s, data)
    model = construct_model(blc, s)
    convolve!(model)
    LL = logpdf.(model, data)
    return sum(LL)
end

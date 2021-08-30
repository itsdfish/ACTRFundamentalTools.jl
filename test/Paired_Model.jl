using SafeTestsets

@safetestset "Paired Model" begin
    using Pkg
    Pkg.activate("../")
    cd(@__DIR__)
    using ACTRModels, Test, Distributions, Random
    include("../models/Paired/Paired_Model.jl")
    include("../models/Paired/Stimuli.jl")
    Random.seed!(552)
    d = 0.5
    parms = (τ = -2.0,noise = true,bll = true,s = 0.5,lf = 0.4,ter = 0.535)
    temp = map(_->simulate(all_stimuli, parms, 8, 20; d), 1:100)
    data = vcat(temp...)
    x = range(d * 0.8, d * 1.2, length = 100)
    y = map(x -> computeLL(data, parms; d=x), x)
    mxv,mxi = findmax(y)
    d′ = x[mxi]
    @test d′ ≈ d atol = 1e-2
end

using SafeTestsets

@safetestset "Count Model" begin
    cd(@__DIR__)
    using Pkg
    Pkg.activate("../")
    using Test, Parameters, FFTDists
    include("../models/Count/Count.jl")

    n_trials = 1000
    s = .3
    blc = 1.5
    data = simulate(n_trials; s, blc)
    filter!(x->x < 2.9, data)
    x = range(blc*.8, blc*1.2, length=50)
    y = map(x->loglike(x, s, data), x)
    mxv,mxi = findmax(y)
    blc′ = x[mxi]
    @test blc′≈ blc atol=5e-2

    x = range(s*.8, s*1.2, length=50)
    y = map(x->loglike(blc, x, data), x)
    mxv,mxi = findmax(y)
    s′ = x[mxi]
    @test s′≈ s atol=5e-2
end

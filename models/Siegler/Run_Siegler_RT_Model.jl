#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using StatsPlots, Revise, ACTRModels, Distributions, Turing, DataFrames
include("Siegler_Model.jl")
include("../../utilities/plotting.jl")
Random.seed!(82210)
#######################################################################################
#                                   Generate Data
#######################################################################################
# mismatch penalty parameter 
δ = 16.0
# retrieval threshold parameter
τ = -.45
# activation noise (logistical scale)
s = .5
parms = (mmp = true,noise = true,mmpFun = simFun,ter = 2.05)
stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
    (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
temp = map(x -> simulate(stimuli, parms; δ, τ, s), 1:5)
data = vcat(vcat(temp...)...)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    δ ~ Normal(16, 8)
    τ ~ Normal(-.45, 1)
    s ~ truncated(Normal(.5, .5), 0.0, Inf)
    data ~ Siegler(δ, τ, s, parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
delta = 0.85
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, delta)
# Start sampling.
chain = sample(model(data, parms), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                      Plot Chains
#######################################################################################
pyplot()
posteriors = plot(chain, seriestype=:density, grid=false, titlefont=font(10), layout=(3,1), 
    size=(250,350), xaxis=font(8), yaxis=font(8))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(stimuli, parms; x...), chain, 1000)
preds = vcat(vcat(preds...)...)
df = DataFrame(preds)
p4 = rt_histogram(df, stimuli)
p5 = response_histogram(df, stimuli)
savefig(p4, "Siegler_RT_Predictions.eps")
savefig(p5, "Siegler_Response_Predictions.eps")
savefig(posteriors, "Siegler_Posteriors.eps")
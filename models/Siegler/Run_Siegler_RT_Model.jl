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
τ = -0.45
# logistic scalar for activation noise
s = 0.5
# fixed parameters 
parms = (mmp = true,noise = true,mmp_fun = sim_fun,ter = 2.05)
# number of blocks
n_blocks = 5
# stimuli for simulation
stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
    (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
# generate data for each block 
temp = map(x -> simulate(stimuli, parms; δ, τ, s), 1:n_blocks)
# flatten data
data = vcat(vcat(temp...)...)
data[1]
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    δ ~ Normal(16, 8)
    τ ~ Normal(-0.45, 1)
    s ~ truncated(Normal(0.5, 0.5), 0.0, Inf)
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
#                                  Posterior Correlations
#######################################################################################
cors = cor(chain)
#######################################################################################
#                                      Plot Chains
#######################################################################################
pyplot()
posteriors = plot(chain, seriestype=:density, grid=false, titlefont=font(10), layout=(3,1), 
    size=(250,350), xaxis=font(8), yaxis=font(8))
plot!(title="", xlabel="δ", subplot=1)
plot!(title="", xlabel="τ", subplot=2)
plot!(title="", xlabel="s", subplot=3)
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(stimuli, parms; x...), chain, 1000)
preds = vcat(vcat(preds...)...)
df = DataFrame(preds)
sort!(stimuli)
p4 = rt_histogram(df, stimuli; size = (300,300))
p5 = response_histogram(df, stimuli; size = (300,300))
savefig(p4, "Siegler_RT_Predictions.eps")
savefig(p5, "Siegler_Response_Predictions.eps")
savefig(posteriors, "Siegler_Posteriors.eps")
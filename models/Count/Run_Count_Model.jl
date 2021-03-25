#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using StatsPlots, Random, ACTRModels, CSV, FFTDists, DifferentialEvolutionMCMC
include("Count.jl")
Random.seed!(761321)
#######################################################################################
#                                   Generate Data
#######################################################################################
n_trials = 50
s = .3
blc = 1.5
data = simulate(n_trials; s, blc)
#######################################################################################
#                                    Define Model
#######################################################################################
priors = (blc=(Normal(1.5, 1),),
  s=(Truncated(Normal(.3, .5), 0.0, Inf),)
  )
bounds = ((-Inf,Inf),(eps(),Inf))
model = DEModel(priors=priors, model=loglike, data=data)
de = DE(priors=priors, bounds=bounds, burnin=1000)
n_iter = 2000
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
println(chains)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
post_plot = plot(chains, xaxis=font(5), yaxis=font(5), seriestype=(:pooleddensity),
  grid=false, titlefont=font(5), size=(200,200), color=:gray)
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x->simulate(10; x...), chains, 1000)
rts = vcat(preds...)
post_pred = histogram(rts, xlabel = "RT", ylabel="Frequency", xaxis=font(7), yaxis=font(7),
    grid=false, color=:grey, leg=false, size=(300,150), titlefont=font(7),
    xlims=(.5,2.5))
savefig(post_plot, "Count_posteriors.eps")
savefig(post_pred, "Count_postpred.eps")

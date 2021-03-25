#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise, StatsPlots, ACTRModels, Distributions, FFTDists, DifferentialEvolutionMCMC
include("Semantic_FFT_Model.jl")
include("model_functions.jl")
include("../../utilities/plotting.jl")
Random.seed!(371401)
#######################################################################################
#                                   Generate Data
#######################################################################################
blc = 1.5
δ = 1.0
parms = (noise = true, τ = 0.0, s = .2, mmp = true)
stimuli = get_stimuli()
n_reps = 10
data = map(s -> simulate(parms, s, n_reps; blc, δ), stimuli)
#######################################################################################
#                                    Define Model
#######################################################################################
priors = (
    blc = (Normal(1.5, 1),),
    δ = (Truncated(Normal(1.0, .5), 0.0, Inf),)
  )
bounds = ((-Inf,Inf),(eps(),Inf))
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
model = DEModel(parms, priors=priors, model=loglike, data=data)
de = DE(bounds=bounds, burnin=1000, priors=priors, n_groups=2, Np=4)
n_iter = 2000
@elapsed chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
#######################################################################################
#                                      Summarize
#######################################################################################
println(chains)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
posteriors = plot(chains, seriestype=:pooleddensity, grid=false, titlefont=font(10),
     xaxis=font(8), yaxis=font(8), color=:grey, size=(300,250))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
rt_preds(s) = posterior_predictive(x -> simulate(parms, s, n_reps; x...), chains, 1000)
temp_preds = map(s -> rt_preds(s), stimuli)
preds = merge.(temp_preds)
posterior_plots = Plots.Plot[]
for (pred, stimulus) in zip(preds, stimuli)
    prob_yes = length(pred.yes)/(length(pred.yes) + length(pred.no))
    object = stimulus.object
    category = stimulus.category
    hist = histogram(layout=(1,2), xlims=(0,2.5),  ylims=(0,3.5), title="$object-$category",
        grid=false, titlefont=font(6), xaxis=font(6), yaxis=font(6), xlabel="Yes RT", 
        xticks = 0:2, yticks=0:3)

    if !isempty(pred.yes)
        histogram!(hist, pred.yes, xlabel="Yes RT", norm=true, grid=false, color=:grey, leg=false, 
            size=(300,250), subplot=1)
        hist[1][1][:y] *= prob_yes
    end

    prob_no = 1 - prob_yes
    object = stimulus.object
    category = stimulus.category
    histogram!(hist, pred.no, xlabel="No RT", norm=true, grid=false, color=:grey, leg=false, 
        size=(300,250), subplot=2)
    hist[2][1][:y] *= prob_no
    push!(posterior_plots, hist)
end
posterior_plot = plot(posterior_plots..., layout=(2,2), size=(330,300))
savefig(posteriors, "Semantic_FFT_Posterior.eps")
savefig(posterior_plot, "Semantic_FFT_Postpred.eps")
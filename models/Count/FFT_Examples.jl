cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise, Distributions, Plots, FFTDists, Plots.Measures, Random
Random.seed!(5505)
###########################################################################
###############################  Define Models  ###########################
###########################################################################
Models = (Gamma(5, .05) + LogNormal(-1, .7),
    Uniform(0, .8) + InverseGaussian(.5, .5),
    Uniform(0, .75) + Uniform(0, .75),
    Uniform(.2, .4) + Exponential(.4) + Normal(.3, .1))
###########################################################################
###############################  Plot Models  #############################
###########################################################################
pyplot()
options = (norm=true, legend=false, ylabel="Density", xlims=(0,2.5), xaxis=font(7),
    yaxis=font(7), grid=false, xticks=[0,1,2])
p = plot(layout = 4, xlims=(0,2.5), margin = 4mm, size=(300,300); options...)
bins = (100,100,20,80)
for (i,model) in enumerate(Models)
    convolve!(model)
    trueVals = rand(model, 10^5)
    histogram!(p, trueVals, color=:grey, bins=bins[i], subplot=i; options...)
    x = 0:.01:2.5
    dens = pdf.(model,x)
    plot!(p, x, dens, color = :black, linewidth = 1.5, subplot=i; options...)
end

savefig(p, "FFT.eps")

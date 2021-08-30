using Revise,Distributed,Plots,MCMCChain
Nchains = 3
addprocs(Nchains)
@everywhere push!(LOAD_PATH,$pwd())
@everywhere push!(LOAD_PATH,$"../../FFT")
@everywhere using Revise,Mamba,PairedAssociatesModel,Convolutions
@everywhere eval(PairedAssociatesModel.extensions)
include("../../Utilities/Utilities.jl")
include("Stimuli.jl")
########################################################################
######################    Generate Data    #############################
########################################################################
Ntrials = 50
stimuli = stimuli[1:5]
Memory = Declarative(;σ=.45,τ=-1.,ϕ=1.0,d=.5)
Data = simulateExperiment(Memory,stimuli,Ntrials)
#λ = s*π/sqrt(3)
@everywhere function constructModel(α,Memory)
  memory=Memory.memory;σ=Memory.σ;ϕ=Memory.ϕ
  normParms = (motor=(μ=.06,N=1),cr=(μ=.05,N=3),att=(μ=.085,N=1))
  μn,σn = convolveNormal(;normParms...)
  model = Normal(μn,σn)+LogNormal(-α+log(ϕ),σ)
  return model
end
data = Dict{Symbol,Any}()
data[:data] = Data
data[:zero] = 0.0
########################################################################
######################    Define Model    ##############################
########################################################################
model = Model(
  zero = Stochastic(
    (d,data) -> Poisson(-logpdf(data,constructModel,d)),false),
  d = Stochastic(() -> Beta(5,5))
)
## Sampling Scheme
scheme = [NUTS([:d])]
#Override default value for target acceptance rate
scheme[1].tune.target = .80
scheme[1].tune.adapt = true
## Sampling Scheme Assignment
setsamplers!(model, scheme)
## Initial Values
inits = [
  Dict{Symbol, Any}(
    :zero => 0,
    :d => rand(Uniform(.3,.7))
  )
  for chains in 1:Nchains
]
########################################################################
######################    Run Model    #################################
########################################################################
@elapsed sim = mcmc(model, data, inits, 2000, burnin=500, thin=1,chains = Nchains)
describe(sim)
diagnostics = Mamba.gelmandiag(sim, mpsrf=true, transform=true)
ch = Chain(sim)
theme(:wong)
p = MCMCChain.plot(ch,xaxis=font(16),yaxis=font(16),seriestype=(:traceplot,:autocorplot,:mixeddensity),
  grid=false)

# pyplot()
# p = MCMCChain.plot(ch,xaxis=font(7),yaxis=font(7),seriestype=(:traceplot),
#   grid=false,size=(300,125),titlefont=font(7))
# p = MCMCChain.plot(ch,xaxis=font(7),yaxis=font(7),seriestype=(:autocorplot),
#   grid=false,size=(300,125),titlefont=font(7))
# p = MCMCChain.plot(ch,xaxis=font(7),yaxis=font(7),seriestype=(:mixeddensity),
#   grid=false,size=(300,125),titlefont=font(7))

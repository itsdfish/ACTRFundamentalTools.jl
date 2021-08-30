using Plots,StatsBase
Nsim = 10^5
Memory = Declarative(;d=.5,τ=-1.0,σ=.45)
c = Chunk(;L=11.0,k=1,N=1)
normParms = (att=(μ=.085,N=1),cr=(μ=.05,N=2))
μ,σ = convolveNormal(;normParms...)
rts = (fixed=fill(0.0,Nsim),variable=fill(0.0,Nsim))
lag = 0.
vals = fill(0.0,Nsim)
for i in 1:Nsim
    c.lags = [lag+μ]
    activation!(c,Memory)
    rts.fixed[i] = rand(LogNormal(-c.act,Memory.σ))
    c.lags = [rand(Normal(lag+μ,σ))]
    activation!(c,Memory)
    vals[i] = c.act
    rts.variable[i] = rand(LogNormal(-c.act,Memory.σ))
end
x = 0:.01:3
v = ecdf(rts.variable)
f = ecdf(rts.fixed)
plot(x,v.(x),grid=false,color=:black,norm=true,leg=false,
    linewidth=2)
plot!(x,f.(x),grid=false,color=:red,norm=true,leg=false,
    linewidth=2)

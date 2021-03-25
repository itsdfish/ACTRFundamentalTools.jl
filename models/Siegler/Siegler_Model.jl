using Parameters, StatsBase, Random
import Distributions: logpdf, rand, loglikelihood

struct Siegler{T1,T2,T3,T4} <: ContinuousUnivariateDistribution
    δ::T1
    τ::T2
    s::T3
    parms::T4
end

Siegler(;δ, τ, s, parms) = Siegler(δ, τ, s, parms)

loglikelihood(d::Siegler, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::Siegler, data::Array{<:NamedTuple,1})
    LL = computeLL(d.parms, data; δ=d.δ, τ=d.τ, s=d.s)
    return LL
end

function simulate(stimuli, parms; args...)
    chunks = populate_memory()
    set_baselevels!(chunks)
    memory = Declarative(;memory=chunks)
    actr = ACTR(;declarative=memory, parms..., args...)
    shuffle!(stimuli)
    N = length(stimuli)
    data = Array{NamedTuple,1}(undef, N)
    for (i,s) in enumerate(stimuli)
        chunk = retrieve(actr; s...)
        rt = compute_RT(actr, chunk) + parms.ter
        if isempty(chunk)
            data[i] = (s...,resp = -100,rt = rt)
        else
            data[i] = (s...,resp = chunk[1].slots.sum,rt = rt)
        end
    end
    return data
end

function simFun(memory, chunk; request...)
    slots = chunk.slots
    p = 0.0; δ = memory.parms.δ
    for (c,v) in request
        p += .1 * δ * abs(slots[c] - v)
    end
    return p
end

function populate_memory(act=0.0)
    chunks = [Chunk(;num1=num1,num2=num2,
        sum=num1 + num2,act=act) for num1 in 0:5
        for num2 in 0:5]
    pop!(chunks)
    return chunks
end

function set_baselevels!(chunks)
    for chunk in chunks
        if chunk.slots.sum < 5
            chunk.bl = .65
        end
    end
    return nothing
end

function computeLL(parms, data; δ, τ, s)
    type = typeof(δ)
    chunks = populate_memory(zero(type))
    set_baselevels!(chunks)
    memory = Declarative(;memory=chunks)
    actr = ACTR(;declarative=memory, parms..., δ, τ, s)
    actr.parms.noise = false
    N = length(chunks) + 1
    @unpack s,ter,τ = actr.parms
    LL = 0.0; idx = 0; μ = Array{type,1}(undef, N)
    σ = s * pi / sqrt(3)
    ϕ = fill(ter, N)
    # retrieval failure case does not have a harvest production
    ϕ[end] -= .05
    for (num1,num2,resp,rt) in data
        compute_activation!(actr; num1=num1, num2=num2)
        map!(x -> x.act, μ, chunks)
        μ[end] = τ
        dist = LNR(;μ=-μ, σ, ϕ)
        if resp != -100 # no retrieval error
            matching = get_chunks(actr; sum=resp)
            log_probs = zeros(type, length(matching))
            for (c,chunk) in enumerate(matching)
                idx = find_index(actr; chunk.slots...)
                log_probs[c] = logpdf(dist, idx, rt)
            end
            LL += logsumexp(log_probs)
        else
            LL += logpdf(dist, N, rt)
        end
    end
    return LL
end

function rt_histogram(df, stimuli)
    vals = NamedTuple[]
    for (num1,num2) in stimuli
        idx = @. ((df[:,:num1] == num1 ) & (df[:,:num2] == num2)) | ((df[:,:num1] == num2) & (df[:,:num2] == num1))
        subdf = df[idx,:]
        str = string(num1, "+", num2)
        temp = filter(x -> x[:resp] != -100, subdf)
        g = groupby(temp, :resp)
        rt_resp = combine(g, :rt => mean)
        push!(vals, (title = str, data = rt_resp))
    end
    p = bar(layout=(2,3), leg=false, xlims=(0,10), xlabel="Response", ylabel="Mean RT",
        size=(300,300), xaxis=font(7), yaxis=font(7), titlefont=font(6), grid=false)
    for (i,v) in enumerate(vals)
        @df v.data bar!(p, :resp, :rt_mean, subplot=i, title=v.title, bar_width=1, color=:grey,
            grid=false, ylims=(0,5.5), xlims=(-.5,9))
    end
    return p
end

function response_histogram(df, stimuli)
    vals = NamedTuple[]
    for (num1,num2) in stimuli
        idx = @. ((df[:,:num1] == num1 ) & (df[:,:num2] == num2)) | ((df[:,:num1] == num2) & (df[:,:num2] == num1))
        subdf = df[idx,:]
        str = string(num1, "+", num2)
        v = filter(x -> x != -100, subdf[:,:resp])
        push!(vals, (title = str, data = v))
    end
    p = histogram(layout=(2,3), leg=false, xlims=(0,10), xlabel="Response",ylabel="Proportion",
        size=(300,300), xaxis=font(7), yaxis=font(6), titlefont=font(6), grid=false)
    for (i,v) in enumerate(vals)
        histogram!(p, v.data, subplot=i, title=v.title, bar_width=1, color=:grey, grid=false,
        normalize=:probability, ylims=(0,1))
    end
    return p
end
module PairedAssociatesModel
    using Distributions,Convolutions
    export computeLogLikelihood,extentions,Chunk,Declarative
    export simulateExperiment,simulateTrial,updateLags!
    export activation!,convolveNormal,getChunk

    extensions = quote
    using Distributions,PairedAssociatesModel
    import Distributions: minimum,maximum,logpdf

      function logpdf(Data,fun,d)
        if (d < 0)
            return -10.0^6
        end
        LL = computeLogLikelihood(d,Data,fun)
        return LL-1000
      end
    end

    mutable struct Chunk
      N::Int
      L::Float64
      created::Float64
      k::Int
      act::Float64
      number::String
      word::String
      recent::Array{Float64,1}
      lags::Array{Float64,1}
    end

    Chunk(;N=1,L=1.0,created=0.0,k=1,act=0.0,number="",word="",recent=Float64[],lags=Float64[])=Chunk(N,L,created,k,act,number,word,recent,lags)

    Broadcast.broadcastable(x::Chunk) = Ref(x)

    mutable struct Declarative{T1,T2,T3,T4}
        memory::Vector{Chunk}
        d::T1
        τ::T2
        σ::T3
        ϕ::T4
    end

    Declarative(;memory=Chunk[],d,τ,σ,ϕ) = Declarative(memory,d,τ,σ,ϕ)

    Broadcast.broadcastable(x::Declarative) = Ref(x)

    function baseLevel(d,lags)
        act = 0.0
        for t in lags
            act += t^-d
        end
        return log(act)
    end

    activation!(Memory) = activation!.(Memory.memory,Memory)

    function activation!(chunk,Memory)
        N = chunk.N; L = chunk.L; k = chunk.k
        d = Memory.d; lags = chunk.lags
        if k == 0
            chunk.act = StandardApprox(N,L,d)
            return nothing
        end
        exact = baseLevel(d,lags)
        approx = 0.0
        if N > k
            tk = lags[k]
            x1 = (N-k)*(L^(1-d)-tk^(1-d))
            x2 = (1-d)*(L-tk)
            approx = x1/x2
        end
        chunk.act = log(exp(exact) + approx)
        return nothing
    end

    function updateRecent!(chunk,curTime)
        k = chunk.k;recent = chunk.recent
        if length(recent) == k
            pop!(recent)
        end
        pushfirst!(recent,curTime)
        return nothing
    end

    function retrievalProb(chunk,Memory)
        τ,σ,=Memory.τ,Memory.σ
        num = exp(chunk.act/σ)
        prob = num/(exp(τ/σ) + num)
        return prob
    end

    """
    Gets necessary data for parameter estimation
    """
    function getChunkInfo(stimulus,Memory,curTime)
        memory = Memory.memory
        N=0;L=0;lag=[0.0]
        if !any(x->Match(x,pairs(stimulus)),memory)
            return N,L,lag
        end
        chunk = getChunk(Memory,(:word,stimulus.word))[1]
        updateLags!(chunk,curTime)
        N=chunk.N; L=chunk.L;lag = chunk.lags
        return N,L,lag
    end

    function retrieve(stimulus,Memory,curTime,deadline)
        #if the chunk is not in memory, there is a retrieval failure
        memory=Memory.memory; d=Memory.d;τ=Memory.τ
        σ=Memory.σ;ϕ=Memory.ϕ
        if !any(x->Match(x,pairs(stimulus)),memory)
            chunk = Chunk(;stimulus...)
            retrievalTime = rand(LogNormal(-τ+log(ϕ),σ))
            retrievalTime = min(retrievalTime,deadline)
            curTime+=retrievalTime
            chunk.created = curTime
            push!(memory,chunk)
            updateRecent!(chunk,curTime)
            return false,retrievalTime
        end
        chunk = getChunk(Memory,(:word,stimulus.word))[1]
        updateLags!(chunk,curTime)
        activation!(chunk,Memory)
        prob = retrievalProb(chunk,Memory)
        if rand() <= (1-prob)
            retrievalTime = rand(LogNormal(-τ+log(ϕ),σ))
            retrievalTime = min(retrievalTime,deadline)
            curTime+=retrievalTime
            return false,retrievalTime
        end
        retrievalTime = rand(LogNormal(-chunk.act+log(ϕ),σ))
        retrievalTime = min(retrievalTime,deadline)
        curTime+=retrievalTime
        if retrievalTime >= deadline
            return (false,retrievalTime)
        end
        updateRecent!.(chunk,curTime)
        return true,retrievalTime
    end

    function updateLags!(chunk,curTime)
        chunk.L = curTime - chunk.created
        chunk.lags = curTime .- chunk.recent
        return nothing
    end

    function updateChunk!(chunk,curTime)
        updateRecent!(chunk,curTime)
        chunk.N += 1
        return nothing
    end

    getChunk(d::Declarative,args::NamedTuple) = getChunk(d.memory,args)

    getChunk(d::Declarative,args...) = getChunk(d.memory,args...)

    getChunk(memory::Vector{Chunk},args::NamedTuple) = getChunk(memory,pairs(args)...)

    function getChunk(memory::Vector{Chunk},args...)
        c = filter(x->Match(x,args),memory)
        return c
    end

    function modify!(c;args...)
        for (k,v) in pairs(args)
            setfield!(c,k,v)
        end
        return nothing
    end

    function Match(chunk,criteria)
        for (c,v) in criteria
            if getfield(chunk,c) != v
                return false
            end
         end
         return true
    end

    function sampleStimulus(stimuli)
        return rand(stimuli)
    end

    function simulateTrial(Memory,stimuli,curTime,isi=5.0)
        #conflict resolution, attend, conflict resolution
        μa = .085; μc = .05
        normParms = (att=(μ=μa,N=1),cr=(μ=μc,N=2))
        feedBackTime = curTime+isi+μa+μc
        stimulus = sampleStimulus(stimuli)
        μ,σ = convolveNormal(;normParms...)
        rt = rand(Normal(μ,σ))
        curTime += rt
        N,L,lag=getChunkInfo(stimulus,Memory,curTime)
        correct,v =retrieve(stimulus,Memory,curTime,isi-rt-.001)
        rt += v
        #Conflict resolution, motor execution
        normParms = (motor=(μ=.06,N=1),cr=(μ=.05,N=1))
        μ,σ = convolveNormal(;normParms...)
        rt += rand(Normal(μ,σ))
        chunk = getChunk(Memory,(:word,stimulus.word))
        updateChunk!.(chunk,feedBackTime)
        data = (stimulus=stimulus,correct=correct,N=N,L=L,lag=lag,rt=rt)
        return data
    end

    function simulateExperiment(Memory,stimuli,Ntrials,isi=5.0)
        data = Array{NamedTuple,1}(undef,Ntrials)
        curTime = 0.0
        for trial in 1:Ntrials
            temp = simulateTrial(Memory,stimuli,curTime,isi)
            data[trial] = temp
            curTime += isi*2
        end
        return data
    end

    function convolveNormal(scaling=2/3;args...)
        #uniform factor
        fact1 = 1/12
        μ,σ = 0.0,0.0
        for (k,v) in pairs(args)
            μ += prod(v)
            σ += fact1.*v.N*(scaling*v.μ)^2
        end
        σ = sqrt(σ)
        return μ,σ
    end

    function computeLogLikelihood(d,Data,fun)
        Memory = Declarative(;σ=.45,τ=-1.,ϕ=1.0,d=d)
        memory=Memory.memory; τ=Memory.τ
        LL = 0.0;p = 0.0;act=0.0
        chunk = Chunk()
        for data in Data
            stimulus = data.stimulus
            if !any(x->Match(x,pairs(stimulus)),memory)
                #probabiilty of false is 1 if no chunk
                p = 1.0
                chunk = Chunk(;stimulus...,k=1)
                act = τ
                push!(memory,chunk)
            else
                chunk = getChunk(Memory,stimulus)[1]
                chunk.N = data.N; chunk.L = data.L
                chunk.lags = data.lag
                activation!(chunk,Memory)
                p = retrievalProb(chunk,Memory)
                data.correct ? p : p = (1-p)
                data.correct ? act=chunk.act : act=τ
            end
            model = fun(act,Memory)
            model.lb=-6;model.ub=6;model.Npoints=2^10
            convolve!(model)
            LL += logpdf(model,data.rt) + log(p)
        end
        return LL
    end
end

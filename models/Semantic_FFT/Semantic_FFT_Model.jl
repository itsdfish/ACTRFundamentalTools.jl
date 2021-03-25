using Parameters, StatsBase, NamedTupleTools

function populate_memory(act=0.0)
    chunks = [
        Chunk(object=:shark, attribute=:dangerous, value=:True, act=act),
        Chunk(object=:shark, attribute=:locomotion, value=:swimming, act=act),
        Chunk(object=:shark, attribute=:category, value=:fish, act=act),
        Chunk(object=:salmon, attribute=:edible, value=:True, act=act),
        Chunk(object=:salmon, attribute=:locomotion, value=:swimming, act=act),
        Chunk(object=:salmon, attribute=:category, value=:fish, act=act),
        Chunk(object=:fish, attribute=:breath, value=:gills, act=act),
        Chunk(object=:fish, attribute=:locomotion, value=:swimming, act=act),
        Chunk(object=:fish, attribute=:category, value=:animal, act=act),
        Chunk(object=:animal, attribute=:moves, value=:True, act=act),
        Chunk(object=:animal, attribute=:skin, value=:True, act=act),
        Chunk(object=:canary, attribute=:color, value=:yellow, act=act),
        Chunk(object=:canary, attribute=:sings, value=:True, act=act),
        Chunk(object=:canary, attribute=:category, value=:bird, act=act),
        Chunk(object=:ostritch, attribute=:flies, value=:False, act=act),
        Chunk(object=:ostritch, attribute=:height, value=:tall, act=act),
        Chunk(object=:ostritch, attribute=:category, value=:bird, act=act),
        Chunk(object=:bird, attribute=:wings, value=:True, act=act),
        Chunk(object=:bird, attribute=:locomotion, value=:flying, act=act),
        Chunk(object=:bird, attribute=:category, value=:animal, act=act),
    ]
    return chunks
end

function simulate(parms, stimulus, n_reps; blc, δ)
    chunks = populate_memory()
    memory = Declarative(;memory=chunks)
    actr = ACTR(;declarative=memory, parms..., blc, δ)
    yes_rts = Float64[]
    no_rts = Float64[]
    for rep in 1:n_reps
        resp,rt = simulate_trial(actr, stimulus)
        resp == :yes ? push!(yes_rts, rt) : push!(no_rts, rt)
    end
    return (stimulus=stimulus, yes_rts = yes_rts, no_rts = no_rts)
end

function simulate_trial(actr, stimulus)
    retrieving = true
    probe = stimulus
    response = :_
    rt = mapreduce(_ -> process_time(.05), +, 1:7)
    rt += mapreduce(_ -> process_time(.085), +, 1:2)
    while retrieving
        rt += process_time(.05)
        chunk = retrieve(actr; object=probe.object, attribute=:category)
        rt += compute_RT(actr, chunk)
        if isempty(chunk)
            rt += process_time(.05) + process_time(.21)
            retrieving = false
            response = :no
        elseif direct_verify(chunk[1], probe)
            rt += process_time(.05) + process_time(.21)
            retrieving = false
            response = :yes
        elseif chain_category(chunk[1], probe)
            probe = delete(probe, :object)
            probe = (object = chunk[1].slots.value, probe...)
        else
            rt += process_time(.05) + process_time(.21)
            retrieving = false
            response = :no
        end
    end
    return response, rt
end

process_time(μ) = rand(Uniform(μ * (2 / 3), μ * (4 / 3)))

function direct_verify(chunk, stim)
    return match(chunk, object=stim.object,
        value=stim.category, attribute=:category)
end

function chain_category(chunk, stim)
    return match(chunk, ==, !=, ==, object=stim.object,
        value=stim.category, attribute=:category)
end

function get_stimuli()
    stimuli = NamedTuple[]
    push!(stimuli, (object = :canary, category = :bird, ans = :yes))
    push!(stimuli, (object = :canary, category = :animal, ans = :yes))
    push!(stimuli, (object = :bird, category = :fish, ans = :no))
    push!(stimuli, (object = :canary, category = :fish, ans = :no))
    return vcat(stimuli...)
end

function zero_chains(actr, stimulus, data)
    LL = zero_chain_yes(actr, stimulus, data.yes_rts)
    LL += zero_chain_no(actr, stimulus, data.no_rts)
    return LL
end

function one_chain(actr, stimulus, data)
    LL = one_chain_yes(actr, stimulus, data.yes_rts)
    LL += one_chain_no(actr, stimulus, data.no_rts)
    return LL
end

function two_chains(actr, stimulus, data)
    return two_chains_no(actr, stimulus, data.no_rts)
end

function zero_chain_yes(actr, stimulus, rts)
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    yes_idx = find_index(actr, object=get_object(stimulus), value=get_category(stimulus))
    retrieval_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=yes_idx)
    model = Normal(μpm, σpm) + retrieval_dist
    convolve!(model)
    LLs = logpdf.(model, rts)
    return sum(LLs)
end

function zero_chain_no(actr, stimulus, rts)
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    yes_idx = find_index(actr, object=get_object(stimulus), value=get_category(stimulus))
    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i in 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != yes_idx
            retrieval_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
            model = Normal(μpm, σpm) + retrieval_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return sum(log.(likelihoods))
end

function one_chain_yes(actr, stimulus, rts)
    probe = stimulus
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 10), visual=(μ = .085,N = 2))
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
    chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx)
    
    probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
    compute_activation!(actr; object=get_object(probe), attribute=:category)
    yes_idx = find_index(actr, object=get_object(probe), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    yes_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=yes_idx)
    model = Normal(μpm, σpm) + chain1_dist + yes_dist
    convolve!(model)
    LLs = logpdf.(model, rts)
    return sum(LLs)
end

function one_chain_no(actr, stimulus, rts)
    likelihoods = one_chain_no_branch1(actr, stimulus, rts)
    likelihoods .+= one_chain_no_branch2(actr, stimulus, rts)
    return sum(log.(likelihoods))
end

function one_chain_no_branch1(actr, stimulus, rts)
    probe = stimulus
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i in 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != chain_idx
            no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
            model = Normal(μpm, σpm) + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

function one_chain_no_branch2(actr, stimulus, rts)
    probe = stimulus
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    # perceptual motor time
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 10), visual=(μ = .085,N = 2))
    # compute activations
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    # get activations
    μ = map(x -> x.act, chunks)
    # add retrieval threshold
    push!(μ, τ)
    # index to chain chunk
    chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
    chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx)
    
    # change probe for second retrieval
    probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
    compute_activation!(actr; object=get_object(probe), attribute=:category)
    yes_idx = find_index(actr, object=get_object(probe), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)

    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i in 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != yes_idx
            no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
            model = Normal(μpm, σpm) + chain1_dist + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

function two_chains_no(actr, stimulus, rts)
    likelihoods = two_chains_no_branch1(actr, stimulus, rts)
    likelihoods .+= two_chains_no_branch2(actr, stimulus, rts)
    likelihoods .+= two_chains_no_branch3(actr, stimulus, rts)
    return sum(log.(likelihoods))
end

function two_chains_no_branch1(actr, stimulus, rts)
    return one_chain_no_branch1(actr, stimulus, rts)
end

function two_chains_no_branch2(actr, stimulus, rts)
    return one_chain_no_branch2(actr, stimulus, rts)
end

function two_chains_no_branch3(actr, stimulus, rts)
    probe = stimulus
    @unpack τ,s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    # perceptual motor time
    μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 11), visual=(μ = .085,N = 2))
    # compute activations
    compute_activation!(actr; object=get_object(stimulus), attribute=:category)
    # get activations
    μ = map(x -> x.act, chunks)
    # add retrieval threshold
    push!(μ, τ)
    # index to chain chunk
    chain_idx1 = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
    chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx1)
    
    # change probe for second retrieval
    probe = (object = get_chunk_value(chunks[chain_idx1]), delete(probe, :object)...)
    compute_activation!(actr; object=get_object(probe), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain2_idx = find_index(actr, object=get_object(probe), attribute=:category)
    chain2_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain2_idx)

    # change probe for third retrieval
    probe = (object = get_chunk_value(chunks[chain2_idx]), delete(probe, :object)...)
    compute_activation!(actr; object=get_object(probe), attribute=:category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain3_idx = find_index(actr, object=get_object(probe), attribute=:category)

    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i in 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != chain3_idx
            no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
            model = Normal(μpm, σpm) + chain1_dist + chain2_dist + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

function loglike(blc, δ, parms, data)
    chunks = populate_memory()
    memory = Declarative(;memory=chunks)
    actr = ACTR(;declarative=memory, parms..., blc=blc, δ=δ, noise=false)
    LL = 0.0
    for d in data
        stimulus = d.stimulus
        if (stimulus.object == :canary) && (stimulus.category == :bird)
            LL += zero_chains(actr, stimulus, d)
        elseif (stimulus.object == :canary) && (stimulus.category == :fish)
            LL += two_chains(actr, stimulus, d)
        else
            LL += one_chain(actr, stimulus, d)
        end
    end
    return LL
end

get_object(x) = x.object
get_category(x) = x.category
get_chunk_value(x) = x.slots.value 

function merge(data)
    yes = map(x->x.yes_rts, data) |> x->vcat(x...)
    no = map(x->x.no_rts, data) |> x->vcat(x...)
    return (yes=yes, no=no)
end
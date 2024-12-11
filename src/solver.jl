@with_kw mutable struct ValidationSolver <: Solver
    agent::PolicyParams # Proposal
    S::AbstractSpace # State space
    N::Int = 1000 # Number of episode samples
    ΔN::Int = 200 # Number of episode samples between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of episode interactions
    a_opt::Union{Nothing, TrainingParams} = nothing# Training parameters for the proposal
    𝒫::NamedTuple = (;) # Extra parameters of the algorithm
    post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    pre_train_callback = (𝒮; kwargs...) -> nothing # callback that gets called once prior to training
    required_columns = Symbol[:traj_importance_weight, :return]
    
    # Stuff specific to estimation
    training_type = :none # Type of training loop
    training_buffer_size = ΔN*max_steps # Whether or not to train on all prior data or just the recent batch
    weight_fn = safe_weight_fn
    agent_pretrain=nothing # Function to pre-train the agent before any rollouts

    # Create buffer to store all of the samples
    buffer_size = N*max_steps
    buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns)
    𝒟 = nothing
end

MCSolver(args...; kwargs...) = ValidationSolver(args...; kwargs...)

function is_estimate(𝒮)
    # Extract the samples
    eps = episodes(𝒮.buffer)
    fs = [𝒮.buffer[:return][1, ep[1]] > 𝒮.𝒫[:f_target][1] for ep in eps]
    ws = [𝒮.buffer[:traj_importance_weight][1, ep[1]] for ep in eps]
    return mean(fs .* ws)
end

function POMDPs.solve(𝒮::ValidationSolver, mdp)
    @assert haskey(𝒮.𝒫, :f_target)
    
    # Pre-train the policy if a function is provided
    if !isnothing(𝒮.agent_pretrain) && 𝒮.i == 0
        𝒮.agent_pretrain(𝒮.agent.π)
        if !isnothing(𝒮.agent.π⁻)
            𝒮.agent.π⁻=deepcopy(𝒮.agent.π)
        end 
    end
    
    # Construct the training buffer
    𝒮.𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.training_buffer_size, device=device(𝒮.agent.π))
    
    # Construct the training buffer, constants, and sampler
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns, max_steps=𝒮.max_steps, traj_weight_fn=𝒮.weight_fn)
    !isnothing(𝒮.log) && isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Info to collect during training
        info = Dict()
        
        # Sample transitions into the batch buffer
        @assert length(𝒮.buffer) < Crux.capacity(𝒮.buffer) # Make sure we never overwrite
        start_index=length(𝒮.buffer) + 1
        𝒮.training_type !=:mc && clear!(𝒮.𝒟)
        episodes!(s, 𝒮.𝒟, store=𝒮.buffer, Neps=𝒮.ΔN, explore=true, i=𝒮.i, cb=(D) -> 𝒮.post_sample_callback(D, info=info, 𝒮=𝒮))
        end_index=length(𝒮.buffer)
        
        # Record the average weight of the samples
        ep_ends = 𝒮.buffer[:episode_end][1,start_index:end_index]
        info[:mean_weight] = sum(𝒮.buffer[:traj_importance_weight][1,start_index:end_index][ep_ends]) / sum(ep_ends)
        @assert !isnan(info[:mean_weight])

        # Log failure rate and pfail estimate
        eps = episodes(𝒮.𝒟)
        fs = [𝒮.𝒟[:return][1, ep[1]] > 𝒮.𝒫[:f_target][1] for ep in eps]
        info[:failure_rate] = sum(fs) / length(fs)
        info[:pfail] = is_estimate(𝒮)

        training_info = Dict()
        if 𝒮.training_type != :none
            
            # Train the proposals
            if 𝒮.training_type == :spais
                training_info = spais_training(𝒮, 𝒮.𝒟)
            else
                @error "uncregonized training type: $training_type"
            end

        end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, training_info,  𝒮=𝒮)
    end
    𝒮.i += 𝒮.ΔN
    
    # Extract the samples
    eps = episodes(𝒮.buffer)
    fs = [𝒮.buffer[:return][1, ep[1]] > 𝒮.𝒫[:f_target][1] for ep in eps]
    ws = [𝒮.buffer[:traj_importance_weight][1, ep[1]] for ep in eps]
    
    fs, ws
end
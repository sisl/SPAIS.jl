function StatebasedProposalAIS(;agent::PolicyParams,
    likelihood::ValidationLikelihood,
    f_target,
    N,
    ΔN,
    max_steps,
    a_opt::NamedTuple=(;), 
    log::NamedTuple=(;), 
    required_columns=[],
    target_ess=nothing,
    name = "spais",
    kwargs...)

    𝒫 = (;prev_buffer=nothing, prev_target_log_ws=nothing, prev_returns=nothing,likelihood=likelihood, f_target=[f_target], target_ess=target_ess)
    required_columns = unique([required_columns..., :logprob, :return, :traj_importance_weight])
    train_params = TrainingParams(
        loss=msa_loss,
        optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Flux.Adam(3f-4)),
        batch_size=256,
        epochs=1,
        a_opt...
    )
    #log=LoggerParams(;dir = "log/$name", period=1000, log...),

    ValidationSolver(;agent,
                     𝒫=𝒫,
                     N=N,
                     ΔN=ΔN,
                     training_buffer_size=ΔN*max_steps,
                     required_columns,
                     training_type=:spais,
                     log=LoggerParams(;dir = "log/$name", log...),
                     a_opt=train_params,
                     kwargs...)

end

function msa_loss(π, 𝒫, 𝒟; info = Dict())
    # Compute the log probability
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    
    # Out of caution, remove any NaNs or Infs
    new_probs = new_probs[.!isnan.(new_probs) .& .!isinf.(new_probs)]

    ignore_derivatives() do
        info[:kl] = mean(𝒟[:logprob][.!isnan.(new_probs) .& .!isinf.(new_probs)] .- new_probs)
    end 

    -mean(new_probs)
end

function get_values_from_buffer(𝒟::ExperienceBuffer, key::Symbol)
    eps = episodes(𝒟)
    values = [𝒟[key][1, ep[1]] for ep in eps]
    return values
end

function update_likelihood(𝒮::ValidationSolver, 𝒟)
    traj_weights = get_values_from_buffer(𝒟, :traj_importance_weight)
    log_ws = log.(traj_weights)
    target_ess = 𝒮.𝒫[:target_ess]
    returns = get_values_from_buffer(𝒟, :return)
    ℓ = 𝒮.𝒫[:likelihood]
    N = 𝒮.ΔN
    objective(t) = ess(exp.(log_ws .+ map(r->logdensity(ℓ, r, t), returns)))/N - target_ess

    try
        res = optimize(objective, 1e-3, ℓ.β)
        return res.minimizer
    catch e
        return ℓ.β
    end
end

function metropolis_hastings_step(𝒮::ValidationSolver, 𝒟; info=Dict())
    likelihood = 𝒮.𝒫[:likelihood]
    prev_buffer = 𝒮.𝒫[:prev_buffer]
    prev_target_log_ws = 𝒮.𝒫[:prev_target_log_ws]
    prev_returns = 𝒮.𝒫[:prev_returns]

    rs = get_values_from_buffer(𝒟, :return)
    ws = get_values_from_buffer(𝒟, :traj_importance_weight)
    ls = map(likelihood, rs)
    target_log_ws = log.(ws) .+ ls

    info[:ess] = ess(exp.(target_log_ws))
    info[:mean_targ_weights] = mean(exp.(target_log_ws))
    info[:std_targ_weights] = std(exp.(target_log_ws))

    if isnothing(prev_buffer)
        𝒮.𝒫 = merge(𝒮.𝒫, (;prev_buffer=deepcopy(𝒟), prev_target_log_ws=target_log_ws, prev_returns=rs))
        return 𝒟
    end

    # If a target effective sample size is specified,
    # update the likelihood temperature (smoothing factor) and the target log weights
    if ~isnothing(𝒮.𝒫[:target_ess])
        if info[:ess]/𝒮.ΔN >= 𝒮.𝒫[:target_ess]
            β_new = update_likelihood(𝒮, 𝒟)
            likelihood.β = β_new

            # update the target log weights with the new likelihood
            target_log_ws = log.(ws) .+ map(likelihood, rs)
            prev_target_log_ws = log.(get_values_from_buffer(prev_buffer, :traj_importance_weight)) .+ map(likelihood, prev_returns)
        end
        info[:temp] = likelihood.β
    end

    # Compute the acceptance probability
    log_acceptance_probs = min.(0.0, target_log_ws .- prev_target_log_ws)

    # Accept or reject the samples
    accept = log.(rand(length(log_acceptance_probs))) .< log_acceptance_probs
    info[:accept_rate] = mean(accept)

    # Construct the new trajectory buffer
    new_buffer = buffer_like(𝒟, capacity=𝒮.training_buffer_size, device=device(𝒮.agent.π))
    new_eps = collect(episodes(𝒟))
    prev_eps = collect(episodes(prev_buffer))

    accepted_eps = get_episodes(𝒮.𝒟, new_eps[accept])
    rejected_eps = get_episodes(prev_buffer, prev_eps[.!accept])
    
    # Check that the accepted_eps and rejected_eps are not empty
    if length(accepted_eps) > 0
        push!(new_buffer, accepted_eps)
    end
    if length(rejected_eps) > 0
        push!(new_buffer, rejected_eps)
    end
    
    # Merge together the target log weights, traj returns and the buffer
    new_target_log_ws = vcat(target_log_ws[accept], prev_target_log_ws[.!accept])
    new_returns = vcat(rs[accept], prev_returns[.!accept])
    𝒮.𝒫 = merge(𝒮.𝒫, (;prev_buffer=new_buffer, prev_target_log_ws=new_target_log_ws, prev_returns=new_returns))

    return new_buffer
end

function spais_training(𝒮::ValidationSolver, 𝒟)
    info = Dict()

    # Perform Metropolis-Hastings step on target distribution
    𝒟 = metropolis_hastings_step(𝒮, 𝒟, info=info)    

    # Train Actor on updated trajectories
    π = 𝒮.agent.π
    batch_train!(actor(π), 𝒮.a_opt, 𝒮.𝒫, deepcopy(𝒟), info=info, π_loss=π)
    
    info
end
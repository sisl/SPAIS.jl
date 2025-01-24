using Revise

using SPAIS
using Random
include("utils.jl")
include("crosswalk.jl")


# Experiment params
Random.seed!(1234)
dir="results/crosswalk"
Neps=50_000
Npretrain=100


# Problem setup params
gt = 5.9e-5
failure_target = 0.99f0
Nsteps_per_episode=100
Px, mdp = gen_crosswalk_problem()
S = state_space(mdp, μ=Float32[0.22, 25.0, -5, 1.5, 1.0, 16.0, 14.5, 1.0, 0.0], σ=Float32[.15f0, .56, 3.0, 0.2, 0.3, 7, 6, 1f0, 1f0])
X = action_space(Px)

# Create a state-based proposal distribution for the pendulum
proposal = statebased_gaussian(S, X)

# Create the SPAIS solver
𝒮 = StatebasedProposalAIS(
    agent=PolicyParams(;π=proposal, pa=Px),
    likelihood=LogisticValidationLikelihood(failure_target, 0.01),
    f_target=failure_target,
    N=Neps,
    ΔN=100,
    max_steps=Nsteps_per_episode,
    S=S,
    agent_pretrain=pretrain_policy(mdp, Px, S, Nepochs=Npretrain),
)

# Run importance sampling to get failures and sample weights
fs, ws = solve(𝒮, mdp)

# Compute the IS estimate of failure probability
pfail = mean(fs .* ws)
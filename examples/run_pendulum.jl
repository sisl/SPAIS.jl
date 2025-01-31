using Revise

using SPAIS
using Random

include("pendulum.jl")
include("utils.jl")

# Experiment params
Random.seed!(1234)
dir="results/pendulum/"
Neps=50_000
Npretrain=100

# Problem setup params
failure_target=π/4
dt = 0.1
Nsteps_per_episode = 20
noise_dist=Normal(0, 0.3)

Px, mdp = gen_topple_mdp(px=noise_dist, Nsteps=Nsteps_per_episode, dt=dt, failure_thresh=failure_target, discrete=false)
S = state_space(mdp; μ=[0.95, 0.0, 0.0], σ=[0.57, 0.1, 0.1])
X = action_space(Px)

# Create a state-based proposal distribution for the pendulum
proposal = statebased_gaussian(S, X)

# Create the SPAIS solver
𝒮 = StatebasedProposalAIS(
    agent=PolicyParams(;π=proposal, pa=Px),
    likelihood=LogisticValidationLikelihood(failure_target, 0.01),
    f_target=failure_target,
    N=Neps,
    ΔN=200,
    max_steps=Nsteps_per_episode,
    S=S,
    agent_pretrain=pretrain_policy(mdp, Px, S, Nepochs=Npretrain),
)

# Run importance sampling to get failures and sample weights
fs, ws = solve(𝒮, mdp)

# Compute the IS estimate of failure probability
pfail = mean(fs .* ws)
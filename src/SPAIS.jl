"""
State-dependent Proposal Adaptive Importance Sampling (SPAIS) algorithm.
"""
module SPAIS

using Reexport
@reexport using Random
@reexport using Flux
@reexport using POMDPs
@reexport using Crux
using Parameters
using Distributions
using LogDensityProblems

using Optim: optimize
using Zygote: ignore_derivatives

include("likelihood.jl")
include("solver.jl")
include("spais.jl")
include("proposals.jl")
include("utils.jl")

export
    ValidationSolver,
    MCSolver,
    StatebasedProposalAIS,
    LogisticValidationLikelihood,
    ExponentialValidationLikelihood,
    statebased_gaussian,
    pretrain_policy





end # module

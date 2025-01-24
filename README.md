# State-dependent Proposal Adaptive Importance Sampling (SPAIS)

[![arXiv](https://img.shields.io/badge/arXiv-2412.02154-b31b1b.svg)](https://arxiv.org/abs/2412.02154)

 Estimate the probability of failure for black-box autonomous systems using State-Dependent Proposal Adaptive Importance Sampling (SPAIS). SPAIS optimizes a state-based proposal distribution to match the optimal importance sampling distribution using Markov score ascent methods.

 ## Installation
```julia
] add https://github.com/sisl/SPAIS.jl
```

## Examples
The `examples/` folder contains scripts for running failure probability estimation on the inverted pendulum problem from the paper.

To run the examples, first run the installation script:

```
cd examples
julia install.jl
```

Next, run the `run_pendulum.jl` script:

```
julia --project run_pendulum.jl
```

Or, try the autonomous vehicle example:

```
julia --project run_crosswalk.jl
```

More examples are coming soon!

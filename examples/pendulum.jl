using POMDPs, POMDPGym, Crux, Distributions, Plots

struct FunPolicy <: Policy
    f
end

Crux.action_space(p::FunPolicy) = ContinuousSpace(1)

function POMDPs.action(p::FunPolicy, s)
    return p.f(s)
end

function continuous_rule(k1, k2, k3)
     (s) -> begin
        ωtarget = sign(s[1])*sqrt(6*10*(1-cos(s[1])))
        -k1*s[1] - k2*s[2] + k3*(ωtarget - s[2])
    end
end

function gen_topple_mdp(;dt=0.2, failure_thresh=π/4, Nsteps=20, xs=[-1f0, -0.25f0, 0f0, 0.25f0, 1f0], px=Normal(0, 0.2), discrete=true)
   maxT = dt*(Nsteps-1)

   env = InvertedPendulumMDP(λcost=0.0f0, failure_thresh=failure_thresh, dt=dt)
   policy = FunPolicy(continuous_rule(0.0, 2.0, -1)) 
   
   if discrete
      probs = pdf.(px, xs)
      Px = DistributionPolicy(DiscreteNonParametric(xs, probs ./ sum(probs)))
   else
      Px = DistributionPolicy(px)
   end
   
   # cost environment
   cost_fn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
   return Px, RMDP(env, policy; cost_fn, added_states=:time, dt, maxT)
end

# Crux.gif(env, simple_policy, "out.gif", max_steps=20, Neps=10)
# heatmap(-3:0.1:3, -8:0.1:8, (x,y) -> action(simple_policy, [x,y])[1])

function plot_trajectories(mdp, π; Neps=100, p=plot(title="$Neps Pendulum Trajectories", xlabel="Timestep", ylabel="θ"), kwargs...)
   D = ExperienceBuffer(episodes!(Sampler(mdp, π), explore=true, Neps=Neps))
   plot_trajectories(D; p, kwargs...)
end

function plot_trajectories(data::ExperienceBuffer; p=plot(xlabel="Timestep", ylabel="θ", title="Pendulum Trajectories"), label="", color=1, alpha=0.3, kwargs...)
   plot!(p, [], color=color, label=label; kwargs...)
   eps = episodes(data)
   for e in eps
       plot!(p, data[:s][2,e[1]:e[2]], label="", alpha=alpha, color=color)
   end
   p
end

function save_trajectories(mdp, π, dir, name; Neps=100, cπs=[], cnames=[], ccolors=[], kwargs...)
   p = plot(title="$Neps Pendulum Trajectories", xlabel="Timestep", ylabel="θ")
   for (pol, n, c) in zip(cπs, cnames, ccolors)
      plot_trajectories(mdp, pol; kwargs..., label=n, Neps, p=p, color=c)
   end
   plot_trajectories(mdp, π; label=name, Neps, p=p, kwargs...)
   savefig("$dir/$(name)_trajectories.png")
   p
end
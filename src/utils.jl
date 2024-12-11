function safe_weight_fn(agent, data, ep)
    logp = trajectory_logpdf(agent.pa, data, ep)
    logq = trajectory_logpdf(agent.π, data, ep)
    if logq == -Inf
        return 0f0
    else
        return exp(logp - logq)
    end	
end

function ess(weights)
    # Compute the effective sample size
    weights = weights ./ sum(weights)
    return 1 / sum(weights .^ 2)
end

function pretrain_policy(mdp, P, S; Ndata=10000, Nepochs=100, batchsize=256)
    (π) -> begin
        # Sample a bunch of data
        D = steps!(Sampler(mdp, P, S=S), explore=true, Nsteps=Ndata)

        # Put it into a flux dataloader
        d = Flux.Data.DataLoader((D[:s], D[:a]), batchsize=batchsize)
        loss(x, y) = -mean(logpdf(π, x, y))
        for i = 1:Nepochs
            Flux.train!(loss, Flux.params(π), d, Adam())
            current_loss = mean([loss(x, y) for (x, y) in d])
            println("Epoch $i, Loss: $current_loss")
        end
    end
end
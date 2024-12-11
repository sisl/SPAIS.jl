using POMDPs, Crux, Flux, BSON, Plots
default(fontfamily="Computer Modern", framestyle=:box)

function experiment_setup(; mdp, Ntrials, dir, plot_init=() -> plot(), gt=nothing)
    try
        mkpath(dir)
    catch
    end

    (𝒮fn, name=nothing) -> begin
        data = Dict(k => [] for k in [:est, :fs, :ws, :gt, :rel_err, :abs_log_err])
        failures = 0
        successes = 0
        while true
            GC.gc()
            S = 𝒮fn()
            try
                fs, ws = solve(S, mdp)
                est = cumsum(fs .* ws) ./ (1:length(fs))
                push!(data[:est], est)
                push!(data[:fs], fs)
                push!(data[:ws], ws)
                if !isnothing(gt)
                    push!(data[:gt], gt)
                    push!(data[:rel_err], abs.(est .- gt) ./ gt)
                    push!(data[:abs_log_err], abs.(log.(est) .- log(gt)))
                end
                successes += 1
                !isnothing(name) && BSON.@save "$dir/$(name)_success_$(successes)_solver.bson" S
            catch e
                println(e)
                failures += 1
                !isnothing(name) && BSON.@save "$dir/$(name)_failure_$(failures)_solver.bson" S
            end
            if successes >= Ntrials || failures >= Ntrials
                break
            end
        end

        if successes > 0 && !isnothing(name)
            BSON.@save "$dir/$name.bson" data
            Neps = length(data[:fs][1])
            plot_init()
            plot!(1:Neps, mean(data[:est]), ribbon=std(data[:est]), label=name)

            savefig("$dir/$name.png")
        end
        return data
    end
end

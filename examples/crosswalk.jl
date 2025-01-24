using AdversarialDriving, AutomotiveSimulator, AutomotiveVisualization
using Distributions, POMDPs, Crux, Random, POMDPTools, POMDPGym, Cairo
using ImageCore
using ImageShow
using Images

struct PointOfClosestApproachMDP{T, A} <: MDP{T, A}
    mdp
    PointOfClosestApproachMDP(mdp::MDP{S,A}) where {S,A} = new{Tuple{Float32, S},A}(mdp)
end

POMDPTools.render(mdp::PointOfClosestApproachMDP, s, args...) = POMDPTools.render(mdp.mdp, s[2], args...)

POMDPs.discount(mdp::PointOfClosestApproachMDP) = 1f0

function POMDPs.initialstate(mdp::PointOfClosestApproachMDP{Tuple{Float32, S}, A}, rng::AbstractRNG = Random.GLOBAL_RNG) where {S, A}
    ImplicitDistribution((rng) -> (0f0, rand(rng, initialstate(mdp.mdp))))
end

function POMDPs.convert_s(::Type{AbstractArray}, s, mdp::PointOfClosestApproachMDP{T, A}) where {T,A}
    Float32.([s[1], convert_s(AbstractArray, s[2], mdp.mdp)...])
end

function POMDPs.gen(mdp::PointOfClosestApproachMDP{Tuple{Float32, S}, A}, s::Tuple{Float32, S}, a, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {S,A}
    sp, r = gen(mdp.mdp, s[2], a, rng)
    biggest_reward = Float32(max(s[1], r))
    if isterminal(mdp.mdp, sp)
        return (;sp=(biggest_reward, sp), r=biggest_reward)
    else
        return (;sp=(biggest_reward, sp), r=0f0)
    end
end

POMDPs.isterminal(mdp::PointOfClosestApproachMDP, s) = isterminal(mdp.mdp, s[2])

struct AdvDrivingAction
    a
end

Base.iterate(v::AdvDrivingAction) = (v, nothing)
Base.iterate(v::AdvDrivingAction, n::Nothing) = nothing

function POMDPs.gen(mdp::MDP{Scene, A}, s::Scene, a::Vector{Float32}, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where A
    ascale=0.25
    a = a .* ascale
    pc = PedestrianControl(da = VecE2(a[1], a[2]), noise=Noise(VecE2(a[3], a[4]), a[5]))
    gen(mdp, s, Disturbance[pc], rng)
end

function gen_good_initialstates(mdp, Nstates, max_target=0.7)
    global GOOD_INITIAL_STATES = []
    adist = product_distribution([Normal(0,1) for i=1:5])
    
    while length(GOOD_INITIAL_STATES) <= Nstates
        
        # Generate challengin scenario
        veh_target = rand(Distributions.Uniform(18,25))
        ped_target = rand(Distributions.Uniform(8,12))

        veh_v = rand(Distributions.Uniform(10, 20))
        ped_v = rand(Distributions.Uniform(0.5, 2.5))

        collision_time = rand(Distributions.Uniform(1,4))

        ped_s = ped_target - ped_v*collision_time
        veh_s = veh_target - (veh_v/2)*collision_time # vehicle is slowing down
        s0=Scene([ez_pedestrian(;id=2, s=ped_s, v=ped_v), ez_ped_vehicle(;id=1, s=veh_s, v=veh_v)])
        
        worst_return = 0f0
        for i=1:15
            s = deepcopy(s0)
            t = 0
            while !isterminal(mdp, s) && t<100
                t+=1
                a = rand(adist)
                ascale=0.5
                a = a .* ascale
                pc = PedestrianControl(da = VecE2(a[1], a[2]), noise=Noise(VecE2(a[3], a[4]), a[5]))
                s, r = gen(mdp, s, Disturbance[pc])
                worst_return = max(worst_return, r)
                if worst_return > max_target
                    break
                end
            end 
            if worst_return > max_target
                break
            end
         end
         if worst_return < max_target
             println("added with max: ", worst_return)
             push!(GOOD_INITIAL_STATES, s0)
         else
             println("rejected with max: ", worst_return)
         end
    end
end

function POMDPTools.render(mdp::MDP{Scene, A}, s, args...) where {A}
    
    frame=AutomotiveVisualization.render([mdp.roadway, s])
    
    tmpfilename = string(tempname(), ".png")
    write(tmpfilename, frame)
    load(tmpfilename)
end

function POMDPs.initialstate(mdp::MDP{Scene, A}, rng::AbstractRNG = Random.GLOBAL_RNG) where A
    # return GOOD_INITIAL_STATES
    
    # Generate challengin scenario
    veh_target = 18.
    veh_start = 0.
    
    ped_target = 8.

    collision_time = 2.
    
    veh_v = 8. #(veh_target - veh_start) / (collision_time)
    ped_v = 1.5

    ped_s = 5. #ped_target - ped_v*collision_time
    
    veh_s = veh_start # vehicle is slowing down
    
    s = Scene([ez_pedestrian(;id=2, s=ped_s, v=ped_v), ez_ped_vehicle(;id=1, s=veh_s, v=veh_v)])
    ImplicitDistribution((rng)->s)
end

function POMDPs.reward(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{Disturbance}, sp::Scene)
    iscollision = length(sp) > 0 && ego_collides(sutid(mdp), sp)
    iscollision ? 1f0 : 20f0 / (Float32(AdversarialDriving.min_dist(s, sutid(mdp)))^2 + 20f0)
end

function POMDPs.isterminal(mdp::MDP{Scene, A}, s::Scene) where A
    isterm_orig = !(sutid(mdp) in s)|| any_collides(s)
    isterm_orig || posf(get_by_id(s, 1)).s > 35 || posf(get_by_id(s, 2)).s > 13
end

function gen_crosswalk_problem()
    Random.seed!(0)
    ## Construct the MDP
    sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=0., v=0.), TIDM(ped_TIDM_template, noisy_observations = true))
    adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=0., v=0.), AdversarialPedestrian())
    mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.2)
    mdp.agents[end].model.idm.v_des=10

    Px = DistributionPolicy(product_distribution([Normal(0,1) for _=1:5]))

    Px, PointOfClosestApproachMDP(mdp)
end


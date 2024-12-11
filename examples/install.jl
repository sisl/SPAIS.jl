using Pkg

Pkg.activate(@__DIR__)

packages = [
    PackageSpec(url="https://github.com/ancorso/POMDPGym.jl"),
    PackageSpec(url="https://github.com/sisl/AdversarialDriving.jl"),
    PackageSpec(url=joinpath(@__DIR__, "..")),
]

ci = haskey(ENV, "CI") && ENV["CI"] == "true"

if ci
    # remove "own" package when on CI
    pop!(packages)
end

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)

Pkg.add("Revise")
Pkg.add("Plots")
Pkg.add("BSON")
Pkg.add("AutomotiveSimulator")
Pkg.add("AutomotiveVisualization")
Pkg.add("Cairo")
Pkg.add("Crux")
Pkg.add("Distributions")
Pkg.add("FileIO")
Pkg.add("Flux")
Pkg.add("ImageCore")
Pkg.add("ImageShow")
Pkg.add("Images")
Pkg.add("JLD2")
Pkg.add("POMDPTools")
Pkg.add("POMDPs")
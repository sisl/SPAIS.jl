function net(S; Nin=S.dims[1], Nout=1, Nhiddens=[64, 32], act=tanh)
    hiddens = [Dense(Nhiddens[idx], Nhiddens[idx+1], relu) for idx in 1:length(Nhiddens)-1]
    Chain(Dense(Nin, Nhiddens[begin], act), hiddens..., Dense(Nhiddens[end], Nout)) # Basic architecture
end

function statebased_gaussian(S, A; Nhiddens=[64, 32])
    base = net(S; Nout=32, Nhiddens=Nhiddens)
    μ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
    logΣ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
    GaussianPolicy(μ, logΣ, true)
end
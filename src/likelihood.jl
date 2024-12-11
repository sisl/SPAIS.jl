abstract type ValidationLikelihood end

# Smooth approximation of the indicator function 
mutable struct LogisticValidationLikelihood <: ValidationLikelihood
    γ::Float64
    β::Float64
end

function (p::LogisticValidationLikelihood)(r)
    c = cdf(Logistic(0.0, p.β), r - p.γ)
    if log(c) == -Inf
        return -1e6
    else
        return log(c)
    end
end

function logdensity(p::LogisticValidationLikelihood, r, β)
    c = cdf(Logistic(0.0, β), r - p.γ)
    if log(c) == -Inf
        return -1e6
    else
        return log(c)
    end
end


mutable struct ExponentialValidationLikelihood <: ValidationLikelihood
    γ::Float64
    β::Float64
end

function (p::ExponentialValidationLikelihood)(r)
    d = p.γ - r
    return logpdf(Exponential(p.β), d)
end

function logdensity(p::ExponentialValidationLikelihood, r, β)
    d = p.γ - r
    return logpdf(Exponential(β), d)

end

import .Soss
import .Soss.TransformVariables as TV
import .Soss.SimpleGraphs as SG

export SossMuseProblem

struct SossMuseProblem{A<:ADTypes.AbstractADType, M<:Soss.AbstractModel, MP<:Soss.AbstractModel} <: AbstractMuseProblem
    autodiff :: A
    model :: M
    model_for_prior :: MP
    xform_z
    xform_θ
    x
    observed_vars
    latent_vars
    hyper_vars
end

@doc doc"""
    SossMuseProblem(model; params, autodiff = AutoForwardDiff())

Specify a MUSE problem with a
[Soss](https://github.com/cscherrer/Soss.jl) model.

The Soss model should be conditioned on the variables which comprise
the "data", and all other variables should be unconditioned. By
default, any parameter which doesn't depend on another parameter will
be estimated by MUSE, but this can be overridden by passing `params`
as a list of symbols. All other non-conditioned and non-`params`
variables will be considered the latent space.

The `autodiff` parameter should be either
`ADTypes.AutoForwardDiff()` or
`ADTypes.AutoZygote()`, specifying which library to use for
automatic differenation.

## Example

```julia
# toy hierarchical problem, using MeasureTheory distributions
funnel = Soss.@model (σ) begin
    θ ~ MeasureTheory.Normal(0, σ)
    z ~ MeasureTheory.Normal(0, exp(θ/2)) ^ 512
    x ~ For(z) do zᵢ
        MeasureTheory.Normal(zᵢ, 1)
    end
end

sim = rand(funnel(3))
model = funnel(3) | (;sim.x)
prob = SossMuseProblem(model)

# get solution
result = muse(prob, (θ=0,))
```

!!! note

    You can also call [`muse`](@ref), etc... directly on the model, e.g.
    `muse(model, (θ=0,))`, in which case the parameter names `params`
    will be read from the keys of provided the starting point.
"""
function SossMuseProblem(
    model::Soss.ConditionalModel; 
    params = leaf_params(model),
    autodiff = ADTypes.AutoForwardDiff()
)
    x = model.obs
    !isempty(x) || error("Model must be conditioned on observed data.")
    sim = rand(model)
    observed_vars = keys(x)
    hyper_vars = (((k, eltype(sim[k])) for k in keys(sim) if k in params)...,)
    latent_vars = keys(delete(sim, (observed_vars..., first.(hyper_vars)...)))
    model_for_prior = Soss.likelihood(Soss.Model(model), first.(hyper_vars)...)(Soss.argvals(model))
    xform_z = Soss.as(model | select(sim, first.(hyper_vars)))
    xform_θ = Soss.as(model | select(sim, latent_vars))
    SossMuseProblem(
        autodiff,
        model,
        model_for_prior,
        xform_z,
        xform_θ,        
        x,
        observed_vars,
        latent_vars,
        hyper_vars
    )
end

function leaf_params(model)
    D = Soss.digraph(Soss.Model(model))
    [x for x in SG.vlist(D) if isempty(filter(!in(keys(Soss.argvals(model))), SG.in_neighbors(D, x)))]
end

function transform_θ(prob::SossMuseProblem, θ)
    TV.inverse(prob.xform_θ, _namedtuple(θ))
end

function inv_transform_θ(prob::SossMuseProblem, θ)
    ComponentVector(TV.transform(prob.xform_θ, θ))
end

function logPriorθ(prob::SossMuseProblem, θ::ComponentVector, ::UnTransformedθ)
    Soss.logdensityof(prob.model_for_prior(_namedtuple(θ)))
end
function logPriorθ(prob::SossMuseProblem, θ::AbstractVector, ::Transformedθ)
    logPriorθ(prob, inv_transform_θ(prob, θ), UnTransformedθ())
end

function ∇θ_logLike(prob::SossMuseProblem, x, z::AbstractVector, θ::ComponentVector, ::UnTransformedθ)
    like = prob.model | (;x..., TV.transform(prob.xform_z, z)...)
    DI.gradient(θ -> Soss.logdensityof(like, _namedtuple(θ)), prob.autodiff, θ)
end
function ∇θ_logLike(prob::SossMuseProblem, x, z::AbstractVector, θ::AbstractVector, ::Transformedθ)
    like = prob.model | (;x..., TV.transform(prob.xform_z, z)...)
    DI.gradient(prob.autodiff, θ) do θ
        Soss.logdensityof(like, _namedtuple(inv_transform_θ(prob, θ)))
    end
end


function logLike_and_∇z_logLike(prob::SossMuseProblem, x, z, θ)
    DI.value_and_gradient(prob.autodiff, z) do z
        Soss.logdensityof(prob.model | (;x..., _namedtuple(θ)...), TV.transform(prob.xform_z, z))
    end
end

function sample_x_z(prob::SossMuseProblem, rng::AbstractRNG, θ)
    sim = Soss.predict(rng, prob.model, _namedtuple(θ))
    x = select(sim, prob.observed_vars)
    z = TV.inverse(prob.xform_z, select(sim, prob.latent_vars))
    (;x, z)
end

# all user-provided θ pass through this first so get it right type /
# order so the user can be lazy
function standardizeθ(prob::SossMuseProblem, θ::NamedTuple)
    Set(keys(θ)) == Set(first.(prob.hyper_vars)) || error("Expected θ to have keys: $(first.(prob.hyper_vars)).")
    ComponentVector((;(k => T.(θ[k]) for (k,T) in prob.hyper_vars)...))
end
function standardizeθ(prob::SossMuseProblem, θ::Number)
    standardizeθ(prob, (;(prob.hyper_vars[1][1] => θ)))
end

_params_from_θ₀(θ₀::Number) = (:θ,)
_params_from_θ₀(θ₀) = keys(θ₀)

function muse!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    muse!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_J!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    get_J!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_H!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    get_H!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end

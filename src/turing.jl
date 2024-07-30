
### Turing interface

import .Turing
import .Turing.DynamicPPL as DynPPL
import ComponentArrays: ComponentVector

export TuringMuseProblem


struct PartialTransformation{T} <: DynPPL.AbstractTransformation 
    transformed_vns :: T
end

function DynPPL.istrans(vi::DynPPL.SimpleVarInfo{NT,T,<:PartialTransformation}, vn::DynPPL.VarName) where {NT,T}
    vn in vi.transformation.transformed_vns
end

function DynPPL.maybe_invlink_before_eval!!(vi::DynPPL.SimpleVarInfo{NT,T,<:PartialTransformation}, context::DynPPL.AbstractContext, model::DynPPL.Model) where {NT,T}
    vi
end


struct TuringMuseProblem{A<:ADTypes.AbstractADType, M<:Turing.Model} <: AbstractMuseProblem
    
    autodiff :: A
    model :: M
    model_for_prior
    trans_z′_θ
    trans_z′_θ′
    vi_θ
    vi_θ′
    x
    observed_vars
    latent_vars
    hyper_vars

end

@doc doc"""

    TuringMuseProblem(model; params, autodiff = Turing.ADBACKEND)

Specify a MUSE problem with a [Turing](https://turing.ml) model.

The Turing model should be conditioned on the variables which comprise
the "data", and all other variables should be unconditioned. By
default, any parameter which doesn't depend on another parameter will
be estimated by MUSE, but this can be overridden by passing `params`
as a list of symbols. All other non-conditioned and non-`params`
variables will be considered the latent space.

The `autodiff` parameter should be either
`ADTypes.AutoForwardDiff()` or
`ADTypes.AutoZygote()`, specifying which library to use for
automatic differenation. The default uses whatever the global
`Turing.ADBACKEND` is currently set to.


## Example

```julia
# toy hierarchical model
Turing.@model function toy()
    σ ~ Uniform(0, 1)
    θ ~ Normal(0, σ)
    z ~ MvNormal(zeros(512), exp(σ/2)*I)
    w ~ MvNormal(z, I)
    x ~ MvNormal(w, I)
    y ~ MvNormal(x, I)
    (;σ, θ, z, w, x, y)
end
sim = toy()()
model = toy() | (;sim.x, sim.y)
prob = TuringMuseProblem(model, params=(:σ, :θ))

# get solution
result = muse(prob, (σ=0.5, θ=0))
```

Here we have chosen `(σ, θ)` to be the parameters which MuseInference
will estimate (note that the default would have only chosen `σ`). The
observed data are `(x,y)` and the remaining `(z,w)` are the latent
space.

!!! note

    You can also call [`muse`](@ref), etc... directly on the model, e.g.
    `muse(model, (σ=0.5, θ=0))`, in which case the parameter names `params`
    will be read from the keys of provided the starting point.

!!! note

    The model function cannot have any of the random variables as 
    arguments, although it can have other parameters as arguments. E.g.,

    ```julia
    # not OK
    @model function toy(x)
        x ~ Normal()
        ...
    end

    # OK
    @model function toy(σ)
        x ~ Normal(σ)
        ...
    end
    ```

"""
function TuringMuseProblem(
    model; 
    params = (:θ,),
    autodiff = nothing,
)

    # set backend based on Turing's by default
    if autodiff == nothing
        if Turing.ADBACKEND[] == :zygote
            autodiff = ADTypes.AutoZygote()
        elseif Turing.ADBACKEND[] == :forwarddiff
            autodiff = ADTypes.AutoForwardDiff()
        else
            error("Unsupposed backend from Turing: $(Turing.ADBACKEND)")
        end
    end
    if (Threads.nthreads() > 1) && (autodiff isa typeof(ADTypes.AutoZygote()))
        error("Turing doesn't support using the Zygote backend when Threads.nthreads()>1. Use a different backend or a single-thread.")
    end

    # ensure tuple
    params = (params...,)
    # prevent this constructor from advancing the default RNG for more clear reproducibility
    rng = copy(Random.default_rng())
    # model is expected to be passed in conditioned on x
    x = ComponentVector(model.context.values)
    # figure out variable names
    observed = keys(x)
    latent = keys(delete(_namedtuple(DynPPL.VarInfo(rng, model)), (observed..., params...)))
    # transform saying that both (z,θ) are transformed
    trans_z′_θ′ = PartialTransformation(_VarName.((latent..., params...)))
    # transform saying only z is transformed
    trans_z′_θ  = PartialTransformation(_VarName.(latent))
    # model with all vars free
    model = DynPPL.decondition(model)
    # model for computing prior, just need any values for (x,z) to condition on here
    vars = _namedtuple(DynPPL.evaluate!!(model, rng)[2])
    model_for_prior = model | select(vars, (observed..., latent...))
    # VarInfos for transforming θ back and forth (can't do this with SimpleVarInfo yet?)
    vi_θ = DynPPL.VarInfo(rng, model_for_prior)
    vi_θ′ = deepcopy(vi_θ)
    DynPPL.settrans!!.((vi_θ′,), true, _VarName.(params))

    TuringMuseProblem(
        autodiff,
        model,
        model_for_prior,
        trans_z′_θ,
        trans_z′_θ′,
        vi_θ,
        vi_θ′,
        x,
        observed,
        latent,
        params
    )

end

function transform_θ(prob::TuringMuseProblem, θ)
    vi = deepcopy(prob.vi_θ)
    DynPPL.setval!(vi, (;θ...))
    DynPPL.link!!(vi, DynPPL.SampleFromPrior(), prob.model)
    ComponentVector(vi)
end

function inv_transform_θ(prob::TuringMuseProblem, θ)
    vi = deepcopy(prob.vi_θ)
    DynPPL.setval!(vi, (;θ...))
    for k in keys(θ)
        DynPPL.settrans!!(vi, true, _VarName(k))
    end
    DynPPL.invlink!!(vi, DynPPL.SampleFromPrior(), prob.model)
    ComponentVector(vi)
end

standardizeθ(prob::TuringMuseProblem, θ::NamedTuple) = 1f0 * ComponentVector(θ) # ensure at least Float32
standardizeθ(prob::TuringMuseProblem, θ::Number) = 
    length(prob.hyper_vars) == 1 ? standardizeθ(prob, (;θ)) : error("Invalid θ type for this problem.")

function logLike(prob::TuringMuseProblem, x, z, θ, θ_space)
    trans = is_transformed(θ_space) ? prob.trans_z′_θ′ : prob.trans_z′_θ
    vi = DynPPL.SimpleVarInfo((;x..., z..., θ...), trans)
    DynPPL.logjoint(prob.model, vi)
end
    
function logPriorθ(prob::TuringMuseProblem, θ, θ_space)
    trans = is_transformed(θ_space) ? prob.trans_z′_θ′ : prob.trans_z′_θ
    vi = DynPPL.SimpleVarInfo((;θ...), trans)
    DynPPL.logprior(prob.model_for_prior, vi)
end

function ∇θ_logLike(prob::TuringMuseProblem, x, z, θ, θ_space)
    DI.gradient(prob.autodiff, θ) do θ
        logLike(prob, x, z, θ, θ_space)
    end
end

function ẑ_at_θ(prob::TuringMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    neglogLike(z) = -logLike(prob, x, z, θ, UnTransformedθ())
    soln = Optim.optimize(optim_only_fg!(neglogLike, prob.autodiff), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    _check_optim_soln(soln)
    soln.minimizer, soln
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = DynPPL.condition(prob.model, θ)
    vi = DynPPL.SimpleVarInfo((;θ...), prob.trans_z′_θ)
    vars = DynPPL.values_as(last(DynPPL.evaluate!!(model, rng, vi)), NamedTuple)
    (x = ComponentVector(select(vars, prob.observed_vars)), z = ComponentVector(select(vars, prob.latent_vars)))
end



# benevolent type-piracy:
function DynPPL.SimpleVarInfo(nt::NamedTuple, trans::DynPPL.AbstractTransformation)
    if isempty(nt)
        T = DynPPL.SIMPLEVARINFO_DEFAULT_ELTYPE
    else
        T = DynPPL.float_type_with_fallback(DynPPL.infer_nested_eltype(typeof(nt)))
    end
    DynPPL.SimpleVarInfo(nt, zero(T), trans)
end

function _namedtuple(vi::DynPPL.VarInfo)
    # `values_as` seems to return Real arays so narrow eltype
    map(x -> identity.(x), DynPPL.values_as(vi, NamedTuple))
end

ComponentVector(vi::DynPPL.VarInfo) = ComponentVector(_namedtuple(vi))

DynPPL.condition(model::DynPPL.Model, x::ComponentVector) = DynPPL.condition(model, (;x...))
_VarName(x::Symbol) = DynPPL.VarName{x}()


_params_from_θ₀(θ₀::Number) = (:θ,)
_params_from_θ₀(θ₀) = keys(θ₀)

function muse!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    muse!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_J!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    get_J!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_H!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    get_H!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end

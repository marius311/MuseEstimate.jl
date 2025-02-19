
### SimpleMuseProblem

struct SimpleMuseProblem{X,S,L,Gθ,Pθ,GZ,A} <: AbstractMuseProblem
    x :: X
    sample_x_z :: S
    logLike :: L
    ∇θ_logLike :: Gθ
    logLike_and_∇z_logLike :: GZ
    logPriorθ :: Pθ
    autodiff :: A
end

@doc doc"""
    SimpleMuseProblem(x, sample_x_z, logLike, logPriorθ=(θ->0); ad=ADTypes.AutoForwardDiff())

Specify a MUSE problem by providing the simulation and posterior
evaluation code by-hand. The argument `x` should be the observed data.
The function `sample_x_z` should have signature:

```julia
function sample_x_z(rng::AbstractRNG, θ)
    # ...
    return (;x, z)
end
```
and return a joint sample of data `x` and latent space `z`. The
function `logLike` should have signature:

```julia
function logLike(x, z, θ) 
    # return log likelihood
end
```

and return the likelihood $\log\mathcal{P}(x,z\,|\,\theta)$ for
your problem. The optional function `logPriorθ` should have signature:

```julia
function logPriorθ(θ)
    # return log prior
end
```

and should return the prior $\log\mathcal{P}(\theta)$ for your
problem. The `autodiff` parameter should be either
`ADTypes.AutoForwardDiff()` or
`ADTypes.AutoZygote()`, specifying which library to use for
automatic differenation through `logLike`.


All variables `(x, z, θ)` can be any types which support basic
arithmetic.

# Example

```julia
# 512-dimensional noisy funnel
prob = SimpleMuseProblem(
    rand(512),
    function sample_x_z(rng, θ)
        z = rand(rng, MvNormal(zeros(512), exp(θ)*I))
        x = rand(rng, MvNormal(z, I))
        (;x, z)
    end,
    function logLike(x, z, θ)
        -(1//2) * (sum((x .- z).^2) + sum(z.^2) / exp(θ) + 512*θ)
    end, 
    function logPrior(θ)
        -θ^2/(2*3^2)
    end;
    autodiff = ADTypes.AutoZygote()
)

# get solution
muse(prob, (θ=1,))
```
"""
function SimpleMuseProblem(x, sample_x_z, logLike, logPriorθ=(θ->0); autodiff::ADTypes.AbstractADType=ADTypes.AutoForwardDiff())
    SimpleMuseProblem(
        x,
        sample_x_z,
        logLike,
        (x,z,θ) -> DI.gradient(θ -> logLike(x,z,θ), autodiff, θ),
        (x,z,θ) -> DI.value_and_gradient(z -> logLike(x,z,θ), autodiff, z),
        logPriorθ,
        autodiff
    )
end

logLike(prob::SimpleMuseProblem, x, z, θ) = prob.logLike(x, z, θ)
∇θ_logLike(prob::SimpleMuseProblem, x, z, θ) = prob.∇θ_logLike(x, z, θ)
logPriorθ(prob::SimpleMuseProblem, θ) = prob.logPriorθ(θ)
logLike_and_∇z_logLike(prob::SimpleMuseProblem, x, z, θ) = prob.logLike_and_∇z_logLike(x, z, θ)
sample_x_z(prob::SimpleMuseProblem, rng::AbstractRNG, θ) = prob.sample_x_z(rng, θ)

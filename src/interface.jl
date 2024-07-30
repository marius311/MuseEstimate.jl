
## interface to be implemented by specific problem types

abstract type AbstractMuseProblem end



struct Transformedθ end
struct UnTransformedθ end
is_transformed(::Transformedθ) = true
is_transformed(::UnTransformedθ) = false


@doc doc"""
    transform_θ(prob::AbstractMuseProblem, θ) 

Map `θ` to a space where its domain is $(-\infty,\infty)$. Defaults to
identity function. 
"""
transform_θ(prob::AbstractMuseProblem, θ) = θ

@doc doc"""
    inv_transform_θ(prob::AbstractMuseProblem, θ) 

Map `θ` from the space where its domain is $(-\infty,\infty)$ back to
the original space. Defaults to identity function.
"""
inv_transform_θ(prob::AbstractMuseProblem, θ) = θ


function logLike end
logLike(prob::AbstractMuseProblem, x, z, θ, θ_space) = 
    logLike(prob::AbstractMuseProblem, x, z, θ)


@doc doc"""
Return the gradient of the joint log likelihood with respect to hyper
parameters `θ`, evaluated at data `x` and latent space `z`. The
signature of the function should be:

    ∇θ_logLike(prob::AbstractMuseProblem, x, z, θ)

If the problem needs a transformation of `θ` to map its domain to
$(-\infty,\infty)$, then it should instead implement:

    ∇θ_logLike(prob::AbstractMuseProblem, x, z, θ, θ_space)

where `θ_space` will be either `Transformedθ()` or `UnTransformedθ()`.
In this case, the `θ` argument will be passed in the space given by
`θ_space` and the gradient should be w.r.t. to `θ` in that space.

`z` must have domain $(-\infty,\infty)$. If a transformation is
required to make this the case, that should be handled internal to
this function and `z` will always refer to the transformed `z`.
"""
function ∇θ_logLike end
∇θ_logLike(prob::AbstractMuseProblem, x, z, θ, θ_space) = 
    ∇θ_logLike(prob::AbstractMuseProblem, x, z, θ)



@doc doc"""
Return a tuple `(logLike, ∇z_logLike)` which give the log likelihood
and its gradient with respect to the latent space `z`, evaluated at
hyper parameters `θ` and data `x` . The signature of the function
should be:

    logLike_and_∇z_logLike(prob::AbstractMuseProblem, x, z, θ)

`z` must have domain $(-\infty,\infty)$. If a transformation is
required to make this the case, that should be handled internal to
this function and `z` will always refer to the transformed `z`.

The `θ` argument to this function will always be in the un-transfored
`θ` space.

!!! note
    
    Alternatively, custom problems can implement `ẑ_at_θ` directly and
    forego this method. The default `ẑ_at_θ` runs LBFGS with Optim.jl
    using `logLike_and_∇z_logLike`.
"""
function logLike_and_∇z_logLike end



@doc doc"""
Return a tuple `(x,z)` with data `x` and latent space `z` which are a
sample from the joint likelihood, given `θ`. The signature of the
function should be:

    sample_x_z(prob::AbstractMuseProblem, rng::AbstractRNG, θ)

Random numbers generated internally should use `rng`.

The `θ` argument to this function will always be in the un-transfored
`θ` space.
"""
function sample_x_z end



@doc doc"""
Return the log-prior at `θ`. The signature of the
function should be:

    logPriorθ(prob::AbstractMuseProblem, θ)

If the problem needs a transformation of `θ` to map its domain to
$(-\infty,\infty)$, then it should instead implement:

    logPriorθ(prob::AbstractMuseProblem, θ, θ_space)

where `θ_space` will be either `Transformedθ()` or `UnTransformedθ()`.
In this case, the `θ` argument will be passed in the space given by
`θ_space`. 

Defaults to zero log-prior.
"""
logPriorθ(prob::AbstractMuseProblem, θ, θ_space) = logPriorθ(prob, θ)
logPriorθ(prob::AbstractMuseProblem, θ) = 0


@doc doc"""
Pre-process a user-provided `θ` into the data-structure used
internally in the computation. E.g. this allow the user to pass a
`NamedTuple` to functions like `muse` or `get_J!` while internally
converting it to a `ComponentVector`. The signature of the function
should be:

    standardizeθ(prob::AbstractMuseProblem, θ)

"""
standardizeθ(prob::AbstractMuseProblem, θ) = θ


@doc doc"""
Return the best-fit latent space `z` given data `x` and parameters
`θ`. The signature of the function should be: 

    ẑ_at_θ(prob::AbstractMuseProblem, x, z₀, θ; ∇z_logLike_atol)

The return value should be `(ẑ, info)` where `info` can be any extra
diagonstic info which will be saved in the MUSE result. 

The `θ` argument to this function will always be in the un-transfored
`θ` space.

The `z₀` should be used as a starting guess for the solution. 
        
`z` must have domain $(-\infty,\infty)$. If a transformation is
required to make this the case, that should be handled internal to
this function, and the return value should refer to the transformed
`z`. 

The default implementation of this method uses
[`logLike_and_∇z_logLike`](@ref) and Optim.jl's LBFGS to iteratively
maximize the log likelihood. Custom problems are free to override this
default if desired, in which case [`logLike_and_∇z_logLike`](@ref)
does not need to be implemented.
"""
function ẑ_at_θ(prob::AbstractMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    soln = optimize(Optim.only_fg(z -> .-logLike_and_∇z_logLike(prob, x, z, θ)), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    _check_optim_soln(soln)
    soln.minimizer, soln
end

function _check_optim_soln(soln)
    Optim.converged(soln) || @warn("MAP solution did not converge within tolerance, result could be erroneous. Try tweaking `θ₀` or `∇z_logLike_atol` arguments to `muse` or fixing model.")
    isfinite(soln.minimum) || @error("MAP solution failed with logjoint(MAP)=$(soln.minimum).")
end



@doc doc"""
    ẑ_guess_from_truth(prob::AbstractMuseProblem, x, z, θ)

Return a guess for the best-fit latent space given data, `x`,
parameters, `θ`, and the true value of latent space in the simulation,
`z`. Used as a starting point for [`ẑ_at_θ`](@ref) when that function
is called on simulations. Typical choices may include `zero(z)` (the
default), `z` itself, or some other heurestic guess.
"""
function ẑ_guess_from_truth(prob::AbstractMuseProblem, x, z, θ)
    zero(z)
end


@doc doc"""
    check_self_consistency(
        prob, 
        θ;
        fdm = central_fdm(3, 1),
        atol = 1e-3,
        rng = Random.default_rng(),
        has_volume_factor = true
    )

Checks the self-consistency of a defined problem at a given `θ`, e.g.
check that `inv_transform_θ(prob, transform_θ(prob, θ)) ≈ θ`, etc...
This is mostly useful as a diagonostic when implementing a new
`AbstractMuseProblem`. 

A random `x` and `z` are sampled from `rng`. Finite differences are
computed using `fdm` and `atol` set the tolerance for `≈`.
`has_volume_factor` determines if the transformation includes the
logdet jacobian in the likelihood.
"""
function check_self_consistency(
    prob, 
    θ;
    fdm = central_fdm(3, 1),
    atol = 1e-2,
    rng = Random.default_rng(),
    has_volume_factor = true
)

    θ = standardizeθ(prob, θ)
    x, z = sample_x_z(prob, rng, θ)
    # volume factor which is added by transformations. dont assume the
    # transformation is AD-able (eg it isnt for Turing)
    J(θ) = has_volume_factor ? FiniteDifferences.jacobian(fdm, θ -> transform_θ(prob, θ), θ)[1] : 1
    V(θ) = has_volume_factor ? logdet(J(θ)) : 0
    ∇θ_V(θ) = has_volume_factor ? FiniteDifferences.grad(fdm, V, θ)[1] : 0
    @testset "Self-consistency" begin
        @test inv_transform_θ(prob, transform_θ(prob, θ)) ≈ θ  atol=atol
        @test logPriorθ(prob, θ, UnTransformedθ()) ≈ logPriorθ(prob, transform_θ(prob, θ), Transformedθ()) .+ V(θ)  atol=atol
        @test ∇θ_logLike(prob, x, z, θ, UnTransformedθ()) ≈ J(θ)' * ∇θ_logLike(prob, x, z, transform_θ(prob, θ), Transformedθ()) .+ ∇θ_V(θ)  atol=atol
    end
end

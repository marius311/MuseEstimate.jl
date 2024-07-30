



### MUSE result

@doc doc"""

Stores the result of a MUSE run. Can be constructed by-hand as
`MuseResult()` and passed to any of the inplace `muse!`, `get_J!`, or
`get_H!`.

Fields:

* `θ` — The estimate of the $\theta$ parameters. 
* `Σ, Σ⁻¹` — The approximate covariance of $\theta$ and its inverse. 
* `H, J` — The $H$ and $J$ matrices which form the covariance (see
  [Millea & Seljak, 2021](https://arxiv.org/abs/2112.09354))
* `gs` — The MAP gradient sims used to compute `J`.
* `Hs` — The jacobian sims used to compute `H`. 
* `dist` — A `Normal` or `MvNormal` built from `θ` and `Σ`, for
  convenience. 
* `history` — Internal diagnostic info from the run. 
* `rng` — RNG used to generate sims for this run (so the same sims can
  be reused if resuming later).
* `time` — Total `Millisecond` wall-time spent computing the result.

"""
Base.@kwdef mutable struct MuseResult
    θ = nothing
    H = nothing
    J = nothing
    Σ⁻¹ = nothing
    Σ = nothing
    dist = nothing
    history = []
    gs = []
    Hs = []
    metadata = Dict()
    rng = nothing
    time = Millisecond(0)
end


function Base.show(io::IO, result::MuseResult)
    _print(μ) = @sprintf("%.4g", μ)
    _print(μ, σ) = @sprintf("%.4g±%.3g", μ, σ)
    print(io, "MuseResult(")
    if result.θ != nothing && result.Σ != nothing
        σ² = result.θ isa AbstractVector ? diag(result.Σ) : result.Σ
        str = sprint(show, _print.(result.θ, sqrt.(σ²)))
    elseif result.θ != nothing
        str = sprint(show, _print.(result.θ))
    else
        str = ""
    end
    print(io, replace(str, "\"" => ""))
    print(io,")")
end

### MUSE solver

@doc doc"""

    muse(prob::AbstractMuseProblem, θ₀; kwargs...)
    muse!(result::MuseResult, prob::AbstractMuseProblem, [θ₀=nothing]; kwargs...)

Run the MUSE estimate. The `muse!` form resumes an existing result. If
the `muse` form is used instead, `θ₀` must give a starting guess for
$\theta$.

See [`MuseResult`](@ref) for description of return value. 

Keyword arguments:

* `rng` — Random number generator to use. Taken from `result.rng` or
  `Random.default_rng()` if not passed. 
* `z₀` — Starting guess for the latent space MAP.
* `maxsteps = 50` — Maximum number of iterations. 
* `θ_rtol = 1e-2` — Error tolerance on $\theta$ relative to its
  standard deviation.
* `∇z_logLike_atol = 1e-2` — Absolute tolerance on the $z$-gradient at
  the MAP solution. 
* `nsims = 100` — Number of simulations. 
* `α = 0.7` — Step size for root-finder. 
* `progress = false` — Show progress bar.
* `pool :: AbstractWorkerPool` — Worker pool for parallelization.
* `regularize = identity` — Apply some regularization after each step. 
* `H⁻¹_like = nothing` — Initial guess for the inverse Jacobian of
  $s^{\rm MUSE}(\theta)$
* `H⁻¹_update` — How to update `H⁻¹_like`. Should be `:sims`,
  `:broyden`, or `:diagonal_broyden`. 
* `broyden_memory = Inf` — How many past steps to keep for Broyden
  updates. 
* `checkpoint_filename = nothing` — Save result to a file after each
  iteration. 
* `get_covariance = false` — Also call `get_H` and `get_J` to get the
  full covariance.
* `save_MAPs = false` — Whether to store the MAP solution at each step
  and for each sim and data into the history (for debugging). Defaults
  to `false` since these are generally high-dimensional and may
  consume lots of memory. Can also pass a function which preprocess
  the MAP before storing it (e.g. `save_MAPs = x -> adapt(Array, x)` to
  convert from a GPU to a CPU array).

"""
muse(args...; kwargs...) = muse!(MuseResult(), args...; kwargs...)

@doc doc"""
See [`muse`](@ref).
"""
function muse!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = nothing;
    rng = nothing,
    z₀ = nothing,
    maxsteps = 50,
    θ_rtol = 1e-1,
    ∇z_logLike_atol = 1e-2,
    nsims = 100,
    α = 0.7,
    progress = false,
    pool = LocalWorkerPool(),
    regularize = identity,
    H⁻¹_like′ = nothing,
    H⁻¹_update = :sims,
    broyden_memory = Inf,
    checkpoint_filename = nothing,
    get_covariance = false,
    save_MAPs = false,
)

    result.rng = rng = @something(rng, result.rng, copy(Random.default_rng()))
    θunreg  = θ  = θ₀ = standardizeθ(prob, @something(result.θ, θ₀))
    θunreg′ = θ′ = transform_θ(prob, θ)
    history = result.history
    
    if save_MAPs == true
        save_MAPs = identity
    elseif save_MAPs == false
        save_MAPs = x -> nothing
    end
    
    if !(α isa Base.Callable)
        α = let α = α
            i -> α
        end
    end

    ẑs = fill(@something(z₀, zero(sample_x_z(prob, copy(rng), θ).z)), nsims+1)
    T = eltype(first(ẑs))

    # set up progress bar
    pbar = progress ? RemoteProgress((maxsteps-length(result.history))*(nsims+1), 0.1, "MUSE: ") : nothing

    try
    
        for i = (length(result.history)+1):maxsteps
            
            t₀ = now()

            if i > 2
                Δθ′ = history[end].θ′ - history[end-1].θ′
                sqrt(-(Δθ′' * history[end].H⁻¹_post′ * Δθ′)) < θ_rtol && break
            end

            # MUSE gradient
            gẑs = pmap(pool, [nothing; split_rng(rng, nsims)], ẑs) do rng, ẑ_prev
                x = (rng == nothing) ? prob.x : sample_x_z(prob, rng, θ).x
                local ẑ, history = ẑ_at_θ(prob, x, ẑ_prev, θ; ∇z_logLike_atol)
                g  = ∇θ_logLike(prob, x, ẑ, θ,  UnTransformedθ())
                g′ = ∇θ_logLike(prob, x, ẑ, θ′, Transformedθ())
                progress && ProgressMeter.next!(pbar)
                (;g, g′, ẑ, history)
            end
            (g_like_dat,    g_like_sims...)    = getindex.(gẑs, :g)
            (g_like_dat′,   g_like_sims′...)   = getindex.(gẑs, :g′)
            (ẑ_history_dat, ẑ_history_sims...) = getindex.(gẑs, :history)
            (ẑ_dat,         ẑ_sims...)         = getindex.(gẑs, :ẑ)
            ẑs = getindex.(gẑs, :ẑ)

            g_like′ = g_like_dat′ .- mean(g_like_sims′)
            g_prior′ = DI.gradient(ADTypes.AutoForwardDiff(), θ′) do θ′
                logPriorθ(prob, θ′, Transformedθ())
            end
            g_post′ = g_like′ .+ g_prior′

            # Jacobian
            h⁻¹_like_sims′ = -1 ./ var(g_like_sims′)
            H⁻¹_like_sims′ = h⁻¹_like_sims′ isa Number ? h⁻¹_like_sims′ : Diagonal(h⁻¹_like_sims′)
            if (H⁻¹_like′ == nothing) || (H⁻¹_update == :sims)
                H⁻¹_like′ = H⁻¹_like_sims′
            elseif i > 2 && (H⁻¹_update in [:broyden, :diagonal_broyden])
                # on subsequent steps, do a Broyden's update using at
                # most the previous `broyden_memory` steps
                j₀ = Int(max(2, i - broyden_memory))
                H⁻¹_like′ = history[j₀-1].H⁻¹_like_sims′
                for j = j₀:i-1
                    Δθ′      = history[j].θ′      - history[j-1].θ′
                    Δg_like′ = history[j].g_like′ - history[j-1].g_like′
                    H⁻¹_like′ = H⁻¹_like′ + ((Δθ′ - H⁻¹_like′ * Δg_like′) / (Δθ′' * H⁻¹_like′ * Δg_like′)) * Δθ′' * H⁻¹_like′
                    if H⁻¹_update == :diagonal_broyden
                        H⁻¹_like′ = Diagonal(H⁻¹_like′)
                    end
                end
            end

            H_prior′ = DI.hessian(ADTypes.AutoForwardDiff(), θ′) do θ′
                logPriorθ(prob, θ′, Transformedθ())
            end
            H⁻¹_post′ = inv(inv(H⁻¹_like′) + H_prior′)

            t = now() - t₀
            push!(
                history, 
                (;
                    θ, θunreg, θ′, θunreg′,
                    g_like_sims,
                    g_like_dat′, g_like_sims′, g_like′, g_prior′, g_post′, 
                    H⁻¹_post′, H_prior′, H⁻¹_like′, H⁻¹_like_sims′, 
                    ẑ_history_dat, ẑ_history_sims, t,
                    ẑ_dat = save_MAPs(ẑ_dat), ẑ_sims = save_MAPs.(ẑ_sims)
                )
            )

            # Newton-Rhapson step
            θunreg′ = θ′ .- T.(α(i)) .* (H⁻¹_post′ * g_post′)
            θunreg  = inv_transform_θ(prob, θunreg′)
            θ′ = regularize(θunreg′)
            θ  = inv_transform_θ(prob, θ′)

            # update these as we go for if we stop early
            result.θ = θunreg
            result.gs = g_like_sims
            result.time += t

            (checkpoint_filename != nothing) && save(checkpoint_filename, "result", result)

        end

    finally

        progress && ProgressMeter.finish!(pbar)
        
    end
    
    if get_covariance
        get_J!(result, prob; rng, nsims, ∇z_logLike_atol)
        get_H!(result, prob; rng, nsims=max(1,nsims÷10), ∇z_logLike_atol)
    end
    result

end


@doc doc"""

    get_H!(result::MuseResult, prob::AbstractMuseProblem, [θ₀=nothing]; kwargs...)

Compute the $H$ matrix, which is part of the MUSE covariance
computation (see [Millea & Seljak,
2021](https://arxiv.org/abs/2112.09354)). 

Positional arguments: 

* `result` — `MuseResult` into which to store result
* `prob` — `AbstractMuseProblem` being solved
* `θ₀` — the `θ` at which to evaluate $H$ (default: `result.θ` if it
  exists, otherwise `θ₀` must be given)

Keyword arguments:

* `z₀` — Starting guess for the latent space MAPs. Defaults to random
  sample from prior.
* `∇z_logLike_atol = 1e-2` — Absolute tolerance on the $z$-gradient at
  the MAP solution. 
* `rng` — Random number generator to use. Taken from `result.rng` or
  `Random.default_rng()` if not passed. 
* `nsims` — How many simulations to average over (default: `10`)
* `pmap` — Parallel map function. 
* `progress` — Show progress bar (default: `false`), 
* `skip_errors` — Ignore any MAP that errors (default: `false`)
* `fdm` — A `FiniteDifferenceMethod` used to compute the finite
  difference Jacobian of AD gradients involved in computing $H$
  (defaults to: `FiniteDifferences.central_fdm(3,1)`)
* `step` — A guess for the finite difference step-size (defaults to
  0.1σ for each parameter using J to estimate σ; for this reason its
  recommended to run `get_J!` before `get_H!`). Is only a guess
  because different choices of `fdm` may adapt this.
* `implicit_diff` — Whether to use experimental implicit
  differentiation, rather than finite differences. Will require 2nd
  order AD through your `logLike` so pay close attention to your
  `prob.autodiff`. Either
  `DifferentiationInterface.SecondOrder(ADTypes.AutoForwardDiff(),
  ADTypes.AutoZygote())` or `ADTypes.AutoForwardDiff()` are recommended
  (default: `false`)

"""
function get_H!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = result.θ;
    fdm :: FiniteDifferenceMethod = central_fdm(3,1),
    ∇z_logLike_atol = 1e-2,
    rng = @something(result.rng, Random.default_rng()),
    nsims = 10, 
    step = nothing, 
    pool = LocalWorkerPool(),
    pmap_over = :auto,
    progress = false,
    skip_errors = false,
    z₀ = nothing,
    implicit_diff = false,
    implicit_diff_H1_is_zero = false,
    implicit_diff_cg_kwargs = (maxiter=100, Pl=I),
)

    θ₀ = standardizeθ(prob, @something(θ₀, result.θ))
    𝟘 = zero(θ₀) * zero(θ₀)' # if θ::ComponentArray, helps keep component labels 
    nsims_existing = length(result.Hs)
    nsims_remaining = nsims - nsims_existing
    (nsims_remaining <= 0) && return
    
    t₀ = now()

    rngs = split_rng(rng, nsims_remaining)

    pbar = progress ? RemoteProgress(nsims_remaining*(1+length(θ₀)), 0.1, "get_H: ") : nothing

    # determine if we parallelize over simulations or over columns of
    # the finite-difference jacobian
    if (pmap_over == :jac || ((pmap_over == :auto) && (length(θ₀) > nsims_remaining)))
        pool_sims, pool_jac = (LocalWorkerPool(), pool)
    else
        pool_sims, pool_jac = (pool, LocalWorkerPool())
    end

    if implicit_diff

        Hs = skipmissing(pmap(pool_sims, rngs) do rng

            try

                (x, z, z_start, ẑ) = remotecall_fetch(pool_jac) do
                    (x, z) = sample_x_z(prob, copy(rng), θ₀)
                    z_start = @something(z₀, ẑ_guess_from_truth(prob, x, z, θ₀))
                    ẑ, = ẑ_at_θ(prob, x, z_start, θ₀, ∇z_logLike_atol=1e-1)
                    pbar == nothing || ProgressMeter.next!(pbar)
                    (x, z, z_start, ẑ)
                end
                T = eltype(z_start)
            
                ad_fwd, ad_rev = if prob.autodiff isa DI.SecondOrder
                    # assume forward-over-reverse is provided
                    DI.outer(prob.autodiff), DI.inner(prob.autodiff)
                else
                    prob.autodiff, prob.autodiff
                end
            
                ## non-implicit-diff term
                H1 = implicit_diff_H1_is_zero ? 𝟘 : copyto!(similar(𝟘), DI.jacobian(ad_fwd, θ₀) do θ
                    local x, = sample_x_z(prob, copy(rng), θ)
                    DI.gradient(ad_rev, θ₀) do θ′ 
                        logLike(prob, x, ẑ, θ′, UnTransformedθ())
                    end
                end)
            
                ## term involving dzMAP/dθ via implicit-diff (w/ conjugate-gradient linear solve)
                dFdθ = DI.jacobian(ad_fwd, θ₀) do θ
                    DI.gradient(ad_rev, ẑ) do z
                        logLike(prob, x, z, θ, UnTransformedθ())
                    end
                end
                dFdθ1 = DI.jacobian(ad_fwd, θ₀) do θ
                    local x, = sample_x_z(prob, copy(rng), θ)
                    DI.gradient(ad_rev, ẑ) do z
                        logLike(prob, x, z, θ₀, UnTransformedθ())
                    end
                end
                # A is the operation of the Hessian of logLike w.r.t. z
                A = LinearMap{T}(length(z_start), isposdef=true, issymmetric=true, ishermitian=true) do w
                    DI.jacobian(ad_fwd, 0) do α
                        DI.gradient(ad_rev, ẑ + α * w) do z
                            logLike(prob, x, z, θ₀, UnTransformedθ())
                        end
                    end
                end
                A⁻¹_dFdθ1 = pmap(pool_jac, eachcol(dFdθ1)) do w 
                    A⁻¹_w = cg(A, w; implicit_diff_cg_kwargs..., log=true)
                    pbar == nothing || ProgressMeter.next!(pbar)
                    A⁻¹_w
                end

                cg_hists = map(last, A⁻¹_dFdθ1)
                H2 = copyto!(similar(𝟘), -(dFdθ' * mapreduce(first, hcat, A⁻¹_dFdθ1)))

                H = H1 + H2
                progress && ProgressMeter.next!(pbar)
                return H, cg_hists
        
            catch err
                if skip_errors && !(err isa InterruptException)
                    @warn err
                    return (missing,)
                else
                    rethrow(err)
                end
            end

        end)

        append!(result.Hs, map(first, Hs))
        append!(get!(() -> [], result.metadata, :implicit_diff_cg_hists), map(last, Hs))

    else

        # default to finite difference step size of 0.1σ with σ roughly
        # estimated from g sims, if we have them
        if step == nothing && !isempty(result.gs)
            step = 0.1 ./ std(result.gs)
        end

        # generate simulations and do initial fit at fiducial, used as
        # starting points for finite difference below
        ẑfids = pmap(pool, rngs) do rngs
            (x, z) = sample_x_z(prob, copy(rng), θ₀)
            z_start = @something(z₀, ẑ_guess_from_truth(prob, x, z, θ₀))
            ẑ, = ẑ_at_θ(prob, x, z_start, θ₀; ∇z_logLike_atol)
            progress && ProgressMeter.next!(pbar)
            ẑ
        end

        # finite difference Jacobian
        append!(result.Hs, skipmissing(pmap(pool_sims, ẑfids, rngs) do ẑ₀, rng
            try
                return first(pjacobian(pool_jac, fdm, θ₀, step; pbar) do θ
                    # sim is generated at θ, MAP and gradient are at fiducial θ₀
                    x, = sample_x_z(prob, copy(rng), θ)
                    ẑ, = ẑ_at_θ(prob, x, ẑ₀, θ₀; ∇z_logLike_atol)
                    ∇θ_logLike(prob, x, ẑ, θ₀, UnTransformedθ())
                end)
            catch err
                if skip_errors && !(err isa InterruptException)
                    @warn err
                    return missing
                else
                    rethrow(err)
                end
            end
        end))

    end
 
    result.H = (θ₀ isa Number) ? mean(first, result.Hs) : (mean(result.Hs) .+ 𝟘)
    result.time += now() - t₀
    finalize_result!(result, prob)

end


@doc doc"""

    get_J!(result::MuseResult, prob::AbstractMuseProblem, [θ₀=nothing]; kwargs...)

Compute the $J$ matrix, which is part of the MUSE covariance
computation (see [Millea & Seljak,
2021](https://arxiv.org/abs/2112.09354)). 

Positional arguments: 

* `result` — `MuseResult` into which to store result
* `prob` — `AbstractMuseProblem` being solved
* `θ₀` — the `θ` at which to evaluate $J$ (default: `result.θ` if it
  exists, otherwise `θ₀` must be given)

Keyword arguments:

* `z₀` — Starting guess for the latent space MAPs. Defaults to random
  sample from prior.
* `∇z_logLike_atol = 1e-2` — Absolute tolerance on the $z$-gradient at
  the MAP solution. 
* `rng` — Random number generator to use. Taken from `result.rng` or
  `Random.default_rng()` if not passed. 
* `nsims` — How many simulations to average over (default: `100`)
* `pmap` — Parallel map function. 
* `progress` — Show progress bar (default: `false`), 
* `skip_errors` — Ignore any MAP that errors (default: `false`)
* `covariance_method` — A `CovarianceEstimator` used to compute $J$
  (default: `SimpleCovariance(corrected=true)`)

"""
function get_J!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = nothing; 
    z₀ = nothing,
    ∇z_logLike_atol = 1e-2,
    rng = @something(result.rng, Random.default_rng()),
    nsims = 100, 
    pool = LocalWorkerPool(),
    progress = false, 
    skip_errors = false,
    covariance_method = SimpleCovariance(corrected=true),
)

    θ₀ = standardizeθ(prob, @something(θ₀, result.θ))
    nsims_existing = length(result.gs)
    nsims_remaining = nsims - nsims_existing

    if nsims_remaining > 0

        pbar = progress ? RemoteProgress(nsims_remaining, 0.1, "get_J: ") : nothing

        rngs = split_rng(rng, nsims)[nsims_existing+1:end]

        append!(result.gs, skipmissing(pmap(pool, rngs) do rng
            try
                (x, z) = sample_x_z(prob, rng, θ₀)
                ẑ₀ = @something(z₀, z)
                ẑ, = ẑ_at_θ(prob, x, ẑ₀, θ₀; ∇z_logLike_atol)
                g = ∇θ_logLike(prob, x, ẑ, θ₀, UnTransformedθ())
                return g
            catch err
                if skip_errors && !(err isa InterruptException)
                    @warn err
                    return missing
                else
                    rethrow(err)
                end
            finally
                progress && ProgressMeter.next!(pbar)
            end
        end))

    end

    result.J = (θ₀ isa Number) ? var(result.gs) : cov(covariance_method, identity.(result.gs))
    finalize_result!(result, prob)

end


function finalize_result!(result::MuseResult, prob::AbstractMuseProblem)
    @unpack H, J, θ = result
    if H != nothing && J != nothing && θ != nothing
        𝟘 = zero(J) # if θ::ComponentArray, helps keep component labels 
        H_prior = -DI.hessian(ADTypes.AutoForwardDiff(), result.θ) do θ
            logPriorθ(prob, θ, UnTransformedθ())
        end
        result.Σ⁻¹ = H' * inv(J) * H + H_prior + 𝟘
        result.Σ = inv(result.Σ⁻¹) + 𝟘
        if length(result.θ) == 1
            result.dist = Normal(result.θ[1], sqrt(result.Σ[1]))
        else
            result.dist = MvNormal(result.θ, Symmetric(Array(result.Σ)))
        end
    end
    result
end

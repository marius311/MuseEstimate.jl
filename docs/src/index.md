# MuseInference.jl

[![](https://img.shields.io/badge/documentation-latest-blue.svg)](https://cosmicmar.com/MuseInference.jl/latest) [![](https://img.shields.io/badge/source-github-blue)](https://github.com/marius311/MuseInference.jl)

[![](https://github.com/marius311/MuseInference.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/marius311/MuseInference.jl/actions/workflows/docs.yml)

The Marginal Unbiased Score Expansion (MUSE) method is a generic tool for hierarchical Bayesian inference. MUSE performs approximate marginalization over arbitrary non-Gaussian and high-dimensional latent spaces, providing Gaussianized constraints on hyper parameters of interest. It is much faster than exact methods like Hamiltonian Monte Carlo (HMC), and requires no user input like many Variational Inference (VI), and Likelihood-Free Inference (LFI) or Simulation-Based Inference (SBI) methods. It excels in high-dimensions, which challenge these other methods. It is approximate, so its results may need to be spot-checked against exact methods, but it is itself exact in asymptotic limit of a large number of data modes contributing to each hyperparameter, or in the limit of Gaussian joint likelihood regardless the number of data modes. For more details, see [Millea & Seljak, 2021](https://arxiv.org/abs/2112.09354).


MUSE works on standard hierarchical problems, where the likelihood is of the form:

```math
\mathcal{P}(x\,|\,\theta) = \int {\rm d}z \, \mathcal{P}(x,z\,|\,\theta)
```

In our notation, $x$ are the observed variables (the "data"), $z$ are unobserved "latent" variables, and $\theta$ are some "hyperparameters" of interest. MUSE is applicable when the goal of the analysis is to estimate the hyperparameters, $\theta$, but otherwise, the latent variables, $z$, do not need to be inferred (only marginalized out via the integral above). 

The only requirements to run MUSE on a particular problem are that forward simulations from $\mathcal{P}(x,z\,|\,\theta)$ can be generated, and gradients of the joint likelihood, $\mathcal{P}(x,z\,|\,\theta)$ with respect to $z$ and $\theta$ can be computed. The marginal likelihood is never required, so MUSE could be considered a form of LFI/SBI. 

# Install

MuseInference.jl is a Julia package for computing the MUSE estimate. To install it, run the following from the Julia package prompt:

```
pkg> add https://github.com/marius311/MuseInference.jl
```

# Example

The easiest way to use MuseInference is with problems defined via the Probabilistic Programming Language, [Turing.jl](https://turing.ml/stable/).

First, load up the packages we'll need:

```@example 1
using MuseInference, Turing
using ADTypes, Dates, LinearAlgebra, Printf, Plots, Random, Zygote
Turing.setadbackend(:zygote)
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
Turing.AdvancedVI.PROGRESS[] = false # hide
Turing.PROGRESS[] = false # hide
nothing # hide
```

As an example, consider the following hierarchical problem, which has the classic [Neal's Funnel](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) problem embedded in it. Neal's funnel is a standard example of a non-Gaussian latent space which HMC struggles to sample efficiently without extra tricks. Specifically, we consider the model defined by:

```math
\begin{aligned}
\theta &\sim {\rm Normal(0,3)} \\ 
z_i &\sim {\rm Normal}(0,\exp(\theta/2)) \\ 
x_i &\sim {\rm Normal}(z_i, 1)
\end{aligned}
```

for $i=1...2048$. This problem can be described by the following Turing model:
```@example 1
@model function funnel()
    θ ~ Normal(0, 3)
    z ~ MvNormal(zeros(2048), exp(θ)*I)
    x ~ MvNormal(z, I)
end
nothing # hide
```

Next, let's choose a true value of $\theta=0$ and generate some simulated data which we'll use as "observations":

```@example 1
Random.seed!(1)
(;x) = rand(funnel() | (θ=0,))
model = funnel() | (;x)
nothing # hide
```

We can run HMC on the problem to compute the "true" answer to compare against:


```@example 1
Random.seed!(2)
sample(model, NUTS(10,0.65,init_ϵ=0.5), 10); # warmup # hide
chain = @time sample(model, NUTS(100, 0.65, init_ϵ=0.5), 500)
nothing # hide
```

We next compute the MUSE estimate for the same problem. To reach the same Monte Carlo error as HMC, the number of MUSE simulations should be the same as the effective sample size of the chain we just ran. This is:

```@example 1 
nsims = round(Int, ess(chain)[:θ,:ess])
```

Running the MUSE estimate, 

```@example 1
muse(model, 0; nsims, get_covariance=true) # warmup # hide
Random.seed!(3)
muse_result = @time muse(model, 0; nsims, get_covariance=true)
nothing # hide
```

Now let's plot the different estimates. In this case, MUSE gives a nearly perfect answer in a fraction of the time.

```@example 1
histogram(collect(chain["θ"][:]), normalize=:pdf, bins=10, label=@sprintf("HMC (%.1f seconds)", chain.info.stop_time - chain.info.start_time))
θs = range(-0.5,0.5,length=1000)
plot!(θs, pdf.(muse_result.dist, θs), label=@sprintf("MUSE (%.1f seconds)", (muse_result.time / Millisecond(1000))), lw=2)
vline!([0], c=:black, ls=:dash, alpha=0.5, label=nothing)
plot!(xlabel="θ", ylabel="P(θ|x)", title="2048-dimensional noisy funnel")
```

The timing[^1] difference is indicative of the speedups over HMC that are possible. These get even more dramatic as we increase dimensionality, which is why MUSE really excels on high-dimensional problems.

[^1]: Julia experts may wonder if the `@time` calls above aren't just mainly timing compilation, but this document is generated with hidden "warmup" calls which ensure that only the runtime is measured.

---

It is also possible to use MuseInference without Turing. The MUSE estimate requires three things:

1. A function which samples from the joint likelihood, $\mathcal{P}(x,z\,|\,\theta)$, with signature:

   ```julia
   function sample_x_z(rng::AbstractRNG, θ)
       # ...
       return (;x, z)
   end
   ```

   where `rng` is an `AbstractRNG` object which should be used when generating random numbers, `θ` are the parameters, and return value should be a named tuple `(;x, z)`. 
   
2. A function which computes the joint likelihood, $\mathcal{P}(x,z\,|\,\theta)$, with signature:

   ```julia
   function logLike(x, z, θ) 
       # return log likelihood
   end
   ```

   A user-specifiable automatic differentiation library will be used to take gradients of this function. 
   
3. A function which computes the prior, $\mathcal{P}(\theta)$, with signature:

   ```julia
   function logPrior(θ)
       # return log prior
   end
   ```

   If none is provided, the prior is assumed uniform. 


In all cases, `x`, `z`, and `θ`, can be of any type which supports basic arithmetic, including scalars, `Vector`s, special vector types like `ComponentArray`s, etc...

We can compute the MUSE estimate for the same funnel problem as above. To do so, first we create a `SimpleMuseProblem` object which specifies the three functions:

```@example 1
prob = SimpleMuseProblem(
    x,
    function sample_x_z(rng, θ)
        z = rand(rng, MvNormal(zeros(2048), exp(θ)*I))
        x = rand(rng, MvNormal(z, I))
        (;x, z)
    end,
    function logLike(x, z, θ)
        -(1//2) * (sum((x .- z).^2) + sum(z.^2) / exp(θ) + 2048*θ)
    end, 
    function logPrior(θ)
        -θ^2/(2*3^2)
    end;
    autodiff = ADTypes.AutoZygote()
)
nothing # hide
```

And compute the estimate:

```@example 1
Random.seed!(3)
muse_result_manual = muse(prob, 0; nsims, get_covariance=true)
nothing # hide
```

This gives the same answer as before:

```@example 1
(muse_result.θ[1], muse_result_manual.θ)
```

---
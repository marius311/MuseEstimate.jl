
_map(args...; _...) = map(args...)

# modified version of https://github.com/JuliaDiff/FiniteDifferences.jl/blob/4d30c4389e06dd2295fd880be57bf58ca8dfc1ce/src/grad.jl#L9
# which allows 
# * specifying the step-size
# * specificying a map function (like pmap instead)
# * (parallel-friendly) progress bar
function pjacobian(f, pool, fdm, x, step; pbar=nothing)
    
    x, from_vec = to_vec(x)
    ẏs = pmap(pool, tuple.(eachindex(x),step)) do (n, step)
        j = fdm(zero(eltype(x)), (step==nothing ? () : step)...) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = copy(first(to_vec(f(from_vec(x)))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
        pbar == nothing || ProgressMeter.next!(pbar)
        return j
    end

    return (hcat(ẏs...), )

end


# ComponentArray constructor is ridiculousy slow, this type piracy
# speeds it up for the case that comes up all the time here where the
# named tuple is not nested
function ComponentArrays.make_carray_args(nt :: NamedTuple{<:Any,<:NTuple{N,Union{Number,Vector}} where N})
    i = 1
    ax = map(nt) do v
        len = length(v)
        s = len==1 ? i : i:i+len-1
        i += len
        s
    end
    vec = reduce(vcat, values(nt))
    (vec, ComponentArrays.Axis(ax))
end

function ComponentArrays.make_carray_args(nt :: NamedTuple{<:Any,<:Tuple{Number} where N})
    ([first(nt)], ComponentArrays.Axis(map(_->1, nt)))
end

_namedtuple(nt::NamedTuple) = nt
function _namedtuple(cv::ComponentVector)
    tp = map(k -> getproperty(cv, k), valkeys(cv))
    unval(::Val{k}) where k = k
    NamedTuple{map(unval,valkeys(cv))}(tp)
end

LinearAlgebra.inv(A::ComponentMatrix{<:Real, <:Symmetric}) = ComponentArray(Matrix(inv(getdata(A))), getaxes(A))

# NamedTupleTools's is broken for Zygote
function select(nt::NamedTuple, ks)
    vals = map(k -> nt[k], ks)
    NamedTuple{ks}(vals)
end

# see https://github.com/JuliaDiff/ForwardDiff.jl/issues/593
function Random.randn!(rng::AbstractRNG, A::Array{<:ForwardDiff.Dual})
    A .= randn!(rng, ForwardDiff.value.(A))
end

# worker pool which just falls back to map
struct LocalWorkerPool <: AbstractWorkerPool end
Distributed.pmap(f, ::LocalWorkerPool, args...) = map(f, args...)
Distributed.remotecall_fetch(f, ::LocalWorkerPool, args...) = f(args...)

# worker pool which is equivalent to passing batch_size to pmap
struct BatchWorkerPool <: AbstractWorkerPool
    pool
    batch_size
end
Distributed.pmap(f, bpool::BatchWorkerPool, args...) = pmap(f, bpool.pool, args...; bpool.batch_size)

# split one rng into a bunch in a way that works with generic RNGs
# does not advance the rng
function split_rng(rng::AbstractRNG, N)
    rng_for_split = copy(rng)
    map(1:N) do i
        Random.seed!(copy(rng), rand(rng_for_split, UInt32))
    end
end

versionof(pkg::Module) = Pkg.dependencies()[Base.PkgId(pkg).uuid].version

# allow using InverseMap as an IterativeSolvers preconditioner
LinearAlgebra.ldiv!(dst::AbstractVector, A::InverseMap, src::AbstractVector) = mul!(dst, A.A, src)
LinearAlgebra.ldiv!(A::InverseMap, vec::AbstractVector) = copyto!(vec, mul!(A.A, vec))

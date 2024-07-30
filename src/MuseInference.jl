module MuseInference

using ADTypes: ADTypes
import DifferentiationInterface as DI
using Base.Iterators: repeated
using ComponentArrays
using CovarianceEstimation
using Dates
using Distributed
using Distributions
using FileIO
using FiniteDifferences
using ForwardDiff
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using Markdown
using NamedTupleTools: merge, delete
using Optim
using Pkg
using Printf
using ProgressMeter
using Random
using Requires
using Setfield
using Statistics
using Test
using TOML
using UnPack

export SimpleMuseProblem, MuseResult, muse, muse!, get_J!, get_H!

include("util.jl")
include("ad.jl")
include("progress.jl")
include("interface.jl")
include("muse.jl")
include("simple.jl")
@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
    TURING_COMPAT = Pkg.Versions.semver_spec(TOML.parsefile(joinpath(@__DIR__,"..","test","Project.toml"))["compat"]["Turing"])
    TURING_VERSION = versionof(Turing)
    if TURING_VERSION in TURING_COMPAT
        include("turing.jl")
    else
        @warn """You have Turing $TURING_VERSION but MuseInference requires a version sem-ver compatible with $TURING_COMPAT. 
        Install a compatible version and restart Julia if you wish to use the MuseInference interface to Turing."""
    end
end
@init @require Soss="8ce77f84-9b61-11e8-39ff-d17a774bf41c" begin
    SOSS_COMPAT = Pkg.Versions.semver_spec(TOML.parsefile(joinpath(@__DIR__,"..","test","Project.toml"))["compat"]["Soss"])
    SOSS_VERSION = versionof(Soss)
    if SOSS_VERSION in SOSS_COMPAT
        include("soss.jl")
    else
        @warn """You have Soss $SOSS_VERSION but MuseInference requires a version sem-ver compatible with $SOSS_COMPAT. 
        Install a compatible version and restart Julia if you wish to use the MuseInference interface to Soss."""
    end
end

end

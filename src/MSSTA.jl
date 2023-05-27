#__precompile__(false) 

module MSSTA

using Distributed

#@everywhere begin
    #using Pkg
    #Pkg.activate(".")
    #Pkg.instantiate()
#end

using ITensors
import ITensors
import ITensors.NDTensors: Tensor, BlockSparseTensor, blockview
using ITensorNetworks

import SparseIR: Fermionic, Bosonic, Statistics
import LinearAlgebra: I
using StaticArrays

using TensorCrossInterpolation
import TensorCrossInterpolation: TensorCI, CachedFunction, TensorCI2
import TensorCrossInterpolation as TCI


function __init__()
end

include("mpsedge.jl")
include("util.jl")
include("quantics.jl")
include("binaryop.jl")
include("mul.jl")
include("mps.jl")
include("fouriertransform.jl")
include("imaginarytime.jl")
include("transformer.jl")
include("qtt.jl")
include("patch.jl")
include("grid.jl")
include("tci.jl")

end

module MSSTA

using ITensors
import ITensors
import ITensors.NDTensors: Tensor, BlockSparseTensor, blockview
using ITensorNetworks

import SparseIR: Fermionic, Bosonic, Statistics
import LinearAlgebra: I
using StaticArrays
import Requires: @require

function __init__()
    @require TensorCrossInterpolation = "b261b2ec-6378-4871-b32e-9173bb050604" include("tci.jl")
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

end

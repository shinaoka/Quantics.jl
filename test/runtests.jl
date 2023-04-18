using Test
using MSSTA

@testset "MSSTA.jl" begin
    include("util.jl")
    include("quantics.jl")
    include("binaryop.jl")
    include("fouriertransform.jl")
    include("imaginarytime.jl")
    include("mul.jl")
    include("mps.jl")
    include("transformer.jl")
    include("qtt.jl")
end

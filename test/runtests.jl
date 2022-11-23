using Test
using MSSTA

@testset "MSSTA.jl" begin
    include("util.jl")
    include("binaryop.jl")
    include("fouriertransform.jl")
    include("imaginarytime.jl")
    include("mul.jl")
    include("matmul.jl")
    include("mps.jl")
end

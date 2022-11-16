using Test
using MSSTA

@testset "MSSTA.jl" begin
    include("util.jl")
    include("arithmetic.jl")
    include("fouriertransform.jl")
    include("imaginarytime.jl")
    include("matmul.jl")
    include("mps.jl")
end

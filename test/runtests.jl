using Test

@testset "MSSTA.jl" begin
    include("util.jl")
    include("quantics.jl")
    include("binaryop.jl")
    include("fouriertransform.jl")
    include("imaginarytime.jl")
    include("mul.jl")
    include("mps.jl")
    include("qtt.jl")
    include("transformer.jl")
    include("grid.jl")
    include("tci.jl")
    include("cubature.jl")
end

nothing

using Test
import MSSTA
using ITensors

@testset "mps.jl" begin @testset "onemps" begin
    nbit = 3
    sites = siteinds("Qubit", nbit)
    M = MSSTA.onemps(Float64, sites)
    @test vec(Array(reduce(*, M), sites)) ≈ ones(2^nbit)
end end

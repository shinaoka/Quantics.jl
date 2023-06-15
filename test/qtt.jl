using Test

using MSSTA
using ITensors
ITensors.disable_warn_order()

@testset "qtt.jl" begin
    @testset "expqtt" begin
        R = 10
        sites = siteinds("Qubit", 10)
        f = MSSTA.expqtt(sites, -1.0)
        f_values = vec(Array(reduce(*, f), reverse(sites)))
        xs = collect(LinRange(0, 1, 2^R + 1)[1:(end - 1)])
        f_values_ref = (x -> exp(-x)).(xs)

        @test f_values ≈ f_values_ref
    end
end

nothing

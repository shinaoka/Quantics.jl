using Test
import Quantics
using Memoize

@testset "cubature.jl" begin
    @testset "adaptive_quadrature" begin
        f(x::Float64) = x^2
        R = 30

        result = Quantics.adaptive_quadrature(Float64, f, R; tolerance=1e-12, maxbonddim=100,
                                           verbosity=1)
        @test isapprox(result, 1.0 / 3.0; atol=1e-9)
    end

    @testset "adaptive_cubature" begin
        #f(x::Vector{Float64})::Float64 = exp(x[1] - x[2])
        #ref = -2 + 1/ℯ + ℯ
        #f(x::Vector{Float64})::Float64 = exp(x[1] + x[2])
        #ref = (ℯ - 1)^2
        #f(x::Vector{Float64})::Float64 = x[1]^2 + x[2]^2
        #ref = -2 + 1/ℯ + ℯ
        f(x::Vector{Float64})::Float64 = x[1]^2 + x[2]^2
        ref = 2.0 / 3.0

        R = 30

        result = Quantics.adaptive_cubature(Float64, 2, f, R; tolerance=1e-12, maxbonddim=100,
                                         verbosity=0)
        @test isapprox(result, ref; atol=1e-8)
    end
end

nothing

using Test
import MSSTA: QuanticsInd, QubitInd, asqubit, asquantics, asqubits, quantics_to_index, index_to_quantics
using ITensors
using StaticArrays
import MSSTA

function _to_ntuple(v::MVector{N,T}) where {N,T}
    return v.data
end

@testset "quantics.jl" begin
    @testset "quantics_to_qubit" begin
        @test asqubit(QuanticsInd{2}(1)) == Tuple(QubitInd.((1, 1)))
        @test asqubit(QuanticsInd{2}(2)) == Tuple(QubitInd.((2, 1)))
        @test asqubit(QuanticsInd{2}(3)) == Tuple(QubitInd.((1, 2)))
        @test asqubit(QuanticsInd{2}(4)) == Tuple(QubitInd.((2, 2)))
        @test asqubits(QuanticsInd{2}.([1, 2, 3, 4])) == QubitInd.([1, 1, 2, 1, 1, 2, 2, 2])
    end

    @testset "qubit_to_quantics" begin
        N = 2
        @test QuanticsInd{N}(asqubits((1, 1))) == QuanticsInd{N}(1)
        @test QuanticsInd{N}(asqubits((2, 1))) == QuanticsInd{N}(2)
        @test QuanticsInd{N}(asqubits((1, 2))) == QuanticsInd{N}(3)
        @test QuanticsInd{N}(asqubits((2, 2))) == QuanticsInd{N}(4)
        @test asquantics(Val(2), QubitInd.([1, 1, 2, 1, 1, 2, 2, 2])) ==
              QuanticsInd{N}.([1, 2, 3, 4])
    end

    @testset "asqubits" begin
        R = 3
        @test asqubits(1, R) == QubitInd.([1, 1, 1])
        @test asqubits(2, R) == QubitInd.([1, 1, 2])
        @test asqubits(3, R) == QubitInd.([1, 2, 1])
        @test asqubits(4, R) == QubitInd.([1, 2, 2])
        @test asqubits(5, R) == QubitInd.([2, 1, 1])
        @test asqubits(6, R) == QubitInd.([2, 1, 2])
        @test asqubits(7, R) == QubitInd.([2, 2, 1])
        @test asqubits(8, R) == QubitInd.([2, 2, 2])
    end

    @testset "quantics_to_index" begin
        # quantics => qubit
        # 1        => (1, 1)
        # 4        => (2, 2)
        # 3        => (1, 2)
        #
        # index = (3, 4)
        D = 2 # Two-dimensional space
        R = 3 # Number of bits along each axis
        @test asqubits(QuanticsInd{D}.([1, 4, 3])) == QubitInd.([1, 1, 2, 2, 1, 2])
        @test quantics_to_index(QuanticsInd{D}.([1, 4, 3])) == (3, 4)
        @test index_to_quantics((3, 4), R) == QuanticsInd{D}.([1, 4, 3])
    end
end

nothing

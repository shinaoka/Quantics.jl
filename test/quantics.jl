using Test
import MSSTA: quantics_to_qubit, qubit_to_quantics, quantics_to_index
using ITensors

@testset "quantics.jl" begin
    @testset "quantics_to_qubit" begin
        @test quantics_to_qubit(Val(2), 1) == (1, 1)
        @test quantics_to_qubit(Val(2), 2) == (1, 2)
        @test quantics_to_qubit(Val(2), 3) == (2, 1)
        @test quantics_to_qubit(Val(2), 4) == (2, 2)
        @test quantics_to_qubit(Val(2), [1, 2, 3, 4]) == [1, 1, 1, 2, 2, 1, 2, 2]
    end

    @testset "qubit_to_quantics" begin
        @test qubit_to_quantics([1, 1]) == 1
        @test qubit_to_quantics([1, 2]) == 2
        @test qubit_to_quantics([2, 1]) == 3
        @test qubit_to_quantics([2, 2]) == 4

        @test qubit_to_quantics(Val(2), [1, 1, 1, 2, 2, 1, 2, 2]) == [1, 2, 3, 4]
    end

    @testset "quantics_to_index" begin
        # quantics => qubit
        # 1        => (1, 1)
        # 4        => (2, 2)
        # 3        => (2, 1)
        #
        # index = (4, 3)
        @test quantics_to_index(Val(2), [1, 4, 3])  == (4, 3)
    end
end

nothing

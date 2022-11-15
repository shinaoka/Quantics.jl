using Test
import MultiScaleSpaceTimes
using ITensors

@testset "util.jl" begin
    @testset "replace_mpo_siteinds!" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        M = MPO(sites, ["Y" for n in 1:nbit])
        #@show sites

        sites2 = [Index(2, "n=$n") for n in 1:nbit]
        MultiScaleSpaceTimes.replace_mpo_siteinds!(M, sites, sites2)

        @test all([!hasind(M[n], sites[n]) for n in 1:nbit])
        @test all([!hasind(M[n], sites[n]') for n in 1:nbit])
        @test all([hasind(M[n], sites2[n]) for n in 1:nbit])
        @test all([hasind(M[n], sites2[n]') for n in 1:nbit])
    end

    @testset "combinesiteinds" begin
        # [s1, (s2,s3), (s4,s5), s6]
        nbit = 6
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:2]
        M = randomMPS(sites; linkdims=2)

        Mc = MultiScaleSpaceTimes.combinesiteinds(M, csites; targetsites=sites[2:5])

        @test length(Mc) == 4
        @test all(dim.(siteinds(Mc)) .== [2, 4, 4, 2])
    end

    @testset "splitsiteind" for nbit in [4, 6]
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:nbit÷2]
        D = 3
        mps = randomMPS(csites; linkdims=D)
        mps_split = MultiScaleSpaceTimes.splitsiteind(mps, sites)
        @test vec(Array(reduce(*, mps_split), sites)) ≈ vec(Array(reduce(*, mps), csites))

        mps_reconst = MultiScaleSpaceTimes.combinesiteinds(mps_split, csites)
        @test vec(Array(reduce(*, mps_reconst), csites)) ≈ vec(Array(reduce(*, mps), csites))
    end

    @testset "linkinds" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        a = randomMPS(sites; linkdims=3)
        l = MultiScaleSpaceTimes._linkinds(a, sites)
        @test all(hastags.(l, "Link"))
        @test length(l) == nbit-1
    end

    @testset "linkinds2" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        a = randomMPS(sites; linkdims=3)
        MultiScaleSpaceTimes.addedges!(a)
        l = MultiScaleSpaceTimes._linkinds(a, sites)
        #@show a
        #@show l
        #@show length(l)
        @test all(hastags.(l, "Link"))
        @test length(l) == nbit+1
    end

    @testset "split_tensor" begin
        nsite = 6
        sites = [Index(2, "Qubit, site=$n") for n in 1:nsite]
        tensor = randomITensor(sites)
        tensors = MultiScaleSpaceTimes.split_tensor(tensor, [sites[1:2], sites[3:4], sites[5:6]])
        @test tensor ≈ reduce(*, tensors)
    end
end
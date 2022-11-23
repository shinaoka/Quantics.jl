using Test
import MSSTA
using ITensors

@testset "util.jl" begin
    @testset "replace_mpo_siteinds!" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        M = MPO(sites, ["Y" for n in 1:nbit])
        #@show sites

        sites2 = [Index(2, "n=$n") for n in 1:nbit]
        MSSTA.replace_mpo_siteinds!(M, sites, sites2)

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

        Mc = MSSTA.combinesiteinds(M, csites; targetsites=sites[2:5])

        @test length(Mc) == 4
        @test all(dim.(siteinds(Mc)) .== [2, 4, 4, 2])
    end

    @testset "splitsiteind" for nbit in [4, 6]
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:(nbit ÷ 2)]
        D = 3
        mps = randomMPS(csites; linkdims=D)
        mps_split = MSSTA.splitsiteind(mps, sites)
        @test vec(Array(reduce(*, mps_split), sites)) ≈ vec(Array(reduce(*, mps), csites))

        mps_reconst = MSSTA.combinesiteinds(mps_split, csites)
        @test vec(Array(reduce(*, mps_reconst), csites)) ≈
              vec(Array(reduce(*, mps), csites))
    end

    @testset "linkinds" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        a = randomMPS(sites; linkdims=3)
        l = MSSTA._linkinds(a, sites)
        @test all(hastags.(l, "Link"))
        @test length(l) == nbit - 1
    end

    @testset "linkinds2" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        a = randomMPS(sites; linkdims=3)
        MSSTA.addedges!(a)
        l = MSSTA._linkinds(a, sites)
        #@show a
        #@show l
        #@show length(l)
        @test all(hastags.(l, "Link"))
        @test length(l) == nbit + 1
    end

    @testset "split_tensor" begin
        nsite = 6
        sites = [Index(2, "Qubit, site=$n") for n in 1:nsite]
        tensor = randomITensor(sites)
        tensors = MSSTA.split_tensor(tensor, [sites[1:2], sites[3:4], sites[5:6]])
        @test tensor ≈ reduce(*, tensors)
    end

    @testset "matchsiteinds_mps" begin
        N = 2
        physdim = 2

        sites = [Index(physdim, "n=$n") for n in 1:(2N)]
        sites_sub = sites[1:2:end]
        M = randomMPS(sites_sub) + randomMPS(sites_sub)

        M_ext = MSSTA.matchsiteinds(M, sites)

        tensor = Array(reduce(*, M), sites_sub)
        tensor_reconst = zeros(Float64, fill(physdim, 2N)...)
        tensor_reconst .= reshape(tensor, size(tensor)..., fill(1, N)...)

        tensor2 = Array(reduce(*, M_ext), sites_sub, sites[2:2:end])
        @test tensor2 ≈ tensor_reconst
    end

    @testset "matchsiteinds_mpo" begin
        N = 2
        physdim = 2

        sites = [Index(physdim, "n=$n") for n in 1:(2N)]
        sites_A = sites[1:2:end]
        sites_B = sites[2:2:end]
        M = randomMPO(sites_A) + randomMPO(sites_A)

        M_ext = MSSTA.matchsiteinds(M, sites)

        tensor_ref = reduce(*, M) * reduce(*, [delta(s, s') for s in sites_B])
        tensor_reconst = reduce(*, M_ext)
        @test tensor_ref ≈ tensor_reconst
    end

    @testset "findallsites_by_tag" begin
        sites = [Index(1, "k=1"), Index(1, "x=1"), Index(1, "k=2")]
        @test MSSTA.findallsites_by_tag(sites; tag="k") == [1, 3]
        @test MSSTA.findallsiteinds_by_tag(sites; tag="k") == [sites[1], sites[3]]

        sites = [Index(1, "k=2"), Index(1, "x=1"), Index(1, "k=1")]
        @test MSSTA.findallsites_by_tag(sites; tag="k") == [3, 1]
        @test MSSTA.findallsiteinds_by_tag(sites; tag="k") == [sites[3], sites[1]]
    end
end

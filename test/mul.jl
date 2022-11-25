using Test
import MSSTA
using ITensors
using ITensorTDVP

@testset "mul.jl" begin
    @testset "_preprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = MSSTA.asMPO(randomMPS(sites1))
        M2 = MSSTA.asMPO(randomMPS(sites2))

        mul = MSSTA.MatrixMultiplier(sitesx, sitesy, sitesz)

        M1, M2 = MSSTA.preprocess(mul, M1, M2)

        flag = true
        for n in 1:N
            flag = flag && hasinds(M1[n], sitesx[n], sitesy[n])
            flag = flag && hasinds(M2[n], sitesy[n], sitesz[n])
        end
        @test flag
    end

    @testset "postprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        mul = MSSTA.MatrixMultiplier(sitesx, sitesy, sitesz)

        links = [Index(1, "Link,l=$l") for l in 0:N]
        M = MPO(N)
        for n in 1:N
            M[n] = randomITensor(links[n], links[n + 1], sitesx[n], sitesz[n])
        end

        M = MSSTA.postprocess(mul, M)

        flag = true
        for n in 1:N
            flag = flag && hasind(M[2 * n - 1], sitesx[n])
            flag = flag && hasind(M[2 * n], sitesz[n])
        end
        @test flag
    end

    @testset "matmul" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        mul = MSSTA.MatrixMultiplier(sitesx, sitesy, sitesz)

        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = MSSTA.asMPO(randomMPS(sites1))
        M2 = MSSTA.asMPO(randomMPS(sites2))

        # preprocess
        M1, M2 = MSSTA.preprocess(mul, M1, M2)

        # MPO-MPO contraction
        M = MSSTA.asMPO(contract(M1, M2; alg="naive"))

        # postprocess
        M = MSSTA.postprocess(mul, M)

        M_mat_reconst = reshape(Array(reduce(*, M), [reverse(sitesx)..., reverse(sitesz)]),
                                2^N, 2^N)

        # Reference data
        M1_mat = reshape(Array(reduce(*, M1), [reverse(sitesx)..., reverse(sitesy)]), 2^N,
                         2^N)
        M2_mat = reshape(Array(reduce(*, M2), [reverse(sitesy)..., reverse(sitesz)]), 2^N,
                         2^N)
        M_mat_ref = M1_mat * M2_mat

        @test M_mat_ref ≈ M_mat_reconst
    end

    @testset "elementwise" begin
        N = 5
        sites = [Index(2, "n=$n") for n in 1:N]
        mul = MSSTA.ElementwiseMultiplier(sites)

        M1_ = randomMPS(sites)
        M2_ = randomMPS(sites)
        M1 = MSSTA.asMPO(M1_)
        M2 = MSSTA.asMPO(M2_)

        # preprocess
        M1, M2 = MSSTA.preprocess(mul, M1, M2)

        # MPO-MPO contraction
        M = MSSTA.asMPO(contract(M1, M2; alg="naive"))

        # postprocess
        M = MSSTA.postprocess(mul, M)

        # Comparison with reference data
        M_reconst = Array(reduce(*, M), sites)
        M1_reconst = Array(reduce(*, M1_), sites)
        M2_reconst = Array(reduce(*, M2_), sites)

        @test M_reconst ≈ M1_reconst .* M2_reconst
    end
end

nothing

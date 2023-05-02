using Test
import MSSTA
using ITensors
using ITensorTDVP

"""
Reconstruct 3D matrix
"""
function _tomat3(a)
    sites = siteinds(a)
    N = length(sites)
    Nreduced = N ÷ 3
    sites_ = [sites[1:3:N]..., sites[2:3:N]..., sites[3:3:N]...]
    return reshape(Array(reduce(*, a), sites_), 2^Nreduced, 2^Nreduced, 2^Nreduced)
end

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

    @testset "elementwisemul" begin
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

    @testset "batchedmatmul" begin
        """
        C(x, z, k) = sum_y A(x, y, k) * B(y, z, k)
        """
        nbit = 2
        D = 2
        sx = [Index(2, "Qubit,x=$n") for n in 1:nbit]
        sy = [Index(2, "Qubit,y=$n") for n in 1:nbit]
        sz = [Index(2, "Qubit,z=$n") for n in 1:nbit]
        sk = [Index(2, "Qubit,k=$n") for n in 1:nbit]

        sites_a = collect(Iterators.flatten(zip(sx, sy, sk)))
        sites_b = collect(Iterators.flatten(zip(sy, sz, sk)))
        sites_c = collect(Iterators.flatten(zip(sx, sz, sk)))

        a = randomMPS(sites_a; linkdims=D)
        b = randomMPS(sites_b; linkdims=D)
        init_c = randomMPS(sites_c; linkdims=D)

        # Reference data
        a_arr = _tomat3(a)
        b_arr = _tomat3(b)
        ab_arr = zeros(Float64, 2^nbit, 2^nbit, 2^nbit)
        for k in 1:(2^nbit)
            ab_arr[:, :, k] .= a_arr[:, :, k] * b_arr[:, :, k]
        end

        ab = MSSTA.automul(a, b; tag_row="x", tag_shared="y", tag_col="z", alg="fit", init=init_c)
        ab_arr_reconst = _tomat3(ab)
        @test ab_arr ≈ ab_arr_reconst
    end
end

nothing

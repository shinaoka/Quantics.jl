using Test
import MSSTA
using ITensors
using ITensorTDVP

function _tomat(a)
    sites = siteinds(a)
    N = length(sites)
    halfN = N ÷ 2
    sites_ = [sites[1:2:N]..., sites[2:2:N]...]
    return reshape(Array(reduce(*, a), sites_), 2^halfN, 2^halfN)
end

function _matmul_xy(a, b; kwargs...)
    length(a) == length(b) || error("Length mismatch")
    all(siteinds(a) .== siteinds(b)) || error("Site indices mismatch")

    nbit = length(a) ÷ 3
    sites = siteinds(a)

    sxy = typeof(sites[1])[]
    sz = typeof(sites[1])[]
    for n in 1:nbit
        push!(sxy, sites[3 * n - 2])
        push!(sxy, sites[3 * n - 1])
        push!(sz, sites[3 * n])
    end

    csites = [Index(4, "csite,n=$n") for n in 1:nbit]
    b_ = MSSTA.combinesiteinds(b, csites; targetsites=sxy)

    t_xy = ITensor[]
    MSSTA.tensors_matmul!(t_xy, a, csites; targetsites=sxy)

    t_z = ITensor[]
    MSSTA.tensors_elementwiseprod!(t_z, a; targetsites=sz)

    tensors = ITensor[]
    sites_wrk = typeof(sites[1])[]
    for n in 1:nbit
        push!(tensors, t_xy[n])
        push!(tensors, t_z[n])
        push!(sites_wrk, csites[n])
        push!(sites_wrk, sz[n])
    end

    mpo = MPO(tensors)
    MSSTA.removeedges!(mpo, sites_wrk)
    ab_ = apply(mpo, b_; kwargs...)

    return MSSTA.splitsiteinds(ab_, sxy; targetcsites=csites)
end

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

@testset "matmul.jl" begin
    @testset "matmul" for _matmul in [MSSTA.matmul, MSSTA.matmul_naive]
        #@testset "matmul" for _matmul in [MSSTA.matmul]
        nbit = 6
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:(nbit ÷ 2)]

        D = 2
        a = randomMPS(sites; linkdims=D)
        b = randomMPS(sites; linkdims=D)
        c = randomMPS(sites; linkdims=D)

        abmat = _tomat(a) * _tomat(b)
        abcmat = abmat * _tomat(c)

        ab = _matmul(a, b)
        @test _tomat(ab) ≈ abmat

        abc = _matmul(ab, c)
        @test _tomat(abc) ≈ abcmat
    end

    @testset "matmul_thru_mpo" begin
        nbit = 6
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:(nbit ÷ 2)]

        D = 2
        a = randomMPS(sites; linkdims=D)
        b = randomMPS(sites; linkdims=D)
        mpo_a = MSSTA.tompo_matmul(a, csites)

        b_ = MSSTA.combinesiteinds(b, csites)
        ab = apply(mpo_a, b_)
        ab = MSSTA.splitsiteind(ab, sites)

        abmat = _tomat(a) * _tomat(b)
        @test _tomat(ab) ≈ abmat
    end

    @testset "elementwiseprod" begin
        nbit = 4
        sites = siteinds("Qubit", nbit)

        D = 2
        a = randomMPS(sites; linkdims=D)
        b = randomMPS(sites; linkdims=D)
        ab = apply(MSSTA.tompo_elementwiseprod(a), b)

        ab_ref = _tomat(a) .* _tomat(b)
        @test _tomat(ab) ≈ ab_ref
    end

    @testset "batchedmatmul" begin
        """
        matmul for x and y axes
        """
        nbit = 4
        D = 4
        sx = [Index(2, "Qubit,x=$n") for n in 1:nbit]
        sy = [Index(2, "Qubit,y=$n") for n in 1:nbit]
        sz = [Index(2, "Qubit,z=$n") for n in 1:nbit]

        sites = typeof(sx[1])[]
        sxy = typeof(sx[1])[]

        for n in 1:nbit
            push!(sites, sx[n])
            push!(sites, sy[n])
            push!(sites, sz[n])
            push!(sxy, sx[n])
            push!(sxy, sy[n])
        end

        a = randomMPS(sites; linkdims=D)
        b = randomMPS(sites; linkdims=D)

        # Reference data
        a_arr = _tomat3(a)
        b_arr = _tomat3(b)
        ab_arr = zeros(Float64, 2^nbit, 2^nbit, 2^nbit)
        for z in 1:(2^nbit)
            ab_arr[:, :, z] .= a_arr[:, :, z] * b_arr[:, :, z]
        end

        ab = _matmul_xy(a, b; alg="fit", nsite=2)
        ab_arr_reconst = _tomat3(ab)
        @test ab_arr ≈ ab_arr_reconst
    end

    @testset "_preprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = MSSTA._convert_to_MPO(randomMPS(sites1))
        M2 = MSSTA._convert_to_MPO(randomMPS(sites2))

        muls = [MSSTA.MatrixMultiplier(sx, sy, sz) for (sx, sy, sz) in zip(sitesx, sitesy, sitesz)]

        for mul in muls
            M1, M2 = MSSTA.preprocess(mul, M1, M2)
        end

        flag = true
        for n in 1:N
            flag = flag && hasinds(M1[n], sitesx[n], sitesy[n])
            flag = flag && hasinds(M2[n], sitesy[n], sitesz[n])
        end
        @test flag
    end

    #==
    @testset "_preprocess_matmul2" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = randomMPS(sites1)
        M2 = randomMPS(sites2)
        tensors1 = ITensors.data(M1)
        tensors2 = ITensors.data(M2)
        MSSTA._preprocess_matmul!(tensors1, tensors2, sites1, sites2, "x", "y", "z")

        flag = true
        for n in 1:N
            flag = flag && hasinds(tensors1[n], sitesx[n], sitesy[n])
            flag = flag && hasinds(tensors2[n], sitesy[n], sitesz[n])
        end
        @test flag
    end
    ==#

    @testset "postprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        muls = [MSSTA.MatrixMultiplier(sx, sy, sz) for (sx, sy, sz) in zip(sitesx, sitesy, sitesz)]

        links = [Index(1, "Link,l=$l") for l in 0:N]
        M = MPO(N)
        for n in 1:N
            M[n] = randomITensor(links[n], links[n+1], sitesx[n], sitesz[n])
        end

        for mul in muls
           M = MSSTA.postprocess(mul, M)
        end

        flag = true
        for n in 1:N
            flag = flag && hasind(M[2*n-1], sitesx[n])
            flag = flag && hasind(M[2*n], sitesz[n])
        end
        @test flag

    end

    @test false
end

using Test
import MSSTA
using ITensors
using LinearAlgebra

@testset "transformer.jl" begin
    @testset "upper_lower_triangle" for upper_or_lower in [:upper, :lower]
        R = 3
        sites = siteinds("Qubit", R)
        trimat = MSSTA.upper_lower_triangle_matrix(sites, 1.0; upper_or_lower=upper_or_lower)
        trimatdata = Array(reduce(*, trimat), [reverse(sites')..., reverse(sites)...])
        trimatdata = reshape(trimatdata, 2^R, 2^R)

        ref = upper_or_lower == :lower ? [Float64(i>j) for i in 1:2^R, j in 1:2^R] : [Float64(i<j) for i in 1:2^R, j in 1:2^R] 

        @test trimatdata ≈ ref
    end

    @testset "cusum" begin
        R = 3
        sites = siteinds("Qubit", R)
        UT = MSSTA.upper_lower_triangle_matrix(sites, 1.0; upper_or_lower=:lower)
        f = MSSTA.expqtt(sites, -1.0)
        f_values = vec(Array(reduce(*, f), reverse(sites)))
        xs = collect(LinRange(0, 1, 2^R+1)[1:end-1])

        g = apply(UT, f)
        g_values = vec(Array(reduce(*, g), reverse(sites)))

        g_values_ref = cumsum(f_values) .- f_values # Second term remove the own values

        @test g_values ≈ g_values_ref
    end

    @testset "flip" for nbit in 2:3, mostsignificantdigit in [:left, :right]
        sites = siteinds("Qubit", nbit)

        g = randomMPS(sites)

        if mostsignificantdigit == :left
            op = MSSTA.flipop(sites; rev_carrydirec=true)
            f = apply(op, g; alg="naive")
            g_reconst = vec(Array(reduce(*, g), reverse(sites)))
            f_reconst = vec(Array(reduce(*, f), reverse(sites)))
        else
            op = MSSTA.flipop(sites; rev_carrydirec=false)
            f = apply(op, g; alg="naive")
            g_reconst = vec(Array(reduce(*, g), sites))
            f_reconst = vec(Array(reduce(*, f), sites))
        end

        f_ref = similar(f_reconst)
        for i in 1:(2^nbit)
            f_ref[i] = g_reconst[mod(2^nbit - (i - 1), 2^nbit) + 1]
        end

        @test f_reconst ≈ f_ref
    end

    @testset "reverseaxis" for bc in [-1, 1]
        nbit = 3

        sites = [Index(2, "x=$x") for x in 1:nbit]

        g = randomMPS(sites)

        f = MSSTA.reverseaxis(g; tag="x", alg="naive", bc=bc)
        g_reconst = vec(Array(reduce(*, g), reverse(sites)))
        f_reconst = vec(Array(reduce(*, f), reverse(sites)))

        f_ref = similar(f_reconst)
        for i in 1:(2^nbit)
            f_ref[i] = g_reconst[mod(2^nbit - (i - 1), 2^nbit) + 1]
        end
        f_ref[1] *= bc

        @test f_reconst ≈ f_ref
    end

    @testset "reverseaxis2" begin
        nbit = 3

        sitesx = [Index(2, "x=$x") for x in 1:nbit]
        sitesy = [Index(2, "y=$y") for y in 1:nbit]

        sites = collect(Iterators.flatten(zip(sitesx, sitesy)))

        g = randomMPS(sites)

        function _reconst(M)
            arr = Array(reduce(*, M), [reverse(sitesx)..., reverse(sitesy)...])
            return reshape(arr, 2^nbit, 2^nbit)
        end

        f = MSSTA.reverseaxis(g; tag="x", alg="naive")
        g_reconst = _reconst(g)
        f_reconst = _reconst(f)

        f_ref = similar(f_reconst)
        for j in 1:(2^nbit), i in 1:(2^nbit)
            f_ref[i, j] = g_reconst[mod(2^nbit - (i - 1), 2^nbit) + 1, j]
        end

        @test f_reconst ≈ f_ref
    end

    @testset "phase_rotation" begin
        nqbit = 3
        xvec = collect(0:(2^nqbit - 1))
        θ = 0.1
        sites = [Index(2, "Qubit,x=$x") for x in 1:nqbit]
        _reconst(x) = vec(Array(reduce(*, x), reverse(sites)))

        f = randomMPS(sites)
        f_vec = _reconst(f)

        ref = exp.(im * θ * xvec) .* f_vec

        @test ref ≈ _reconst(MSSTA.phase_rotation(f, θ; tag="x"))
        @test ref ≈ _reconst(MSSTA.phase_rotation(f, θ; targetsites=sites))
    end

    @testset "asdiagonal" begin
        R = 2
        sites = siteinds("Qubit", R)
        sites′ = [Index(2, "Qubit,n′=$n") for n in 1:R]

        M = randomMPS(sites)

        for which_new in ["left", "right"]
            Mnew = MSSTA.asdiagonal(M, sites′; tag="n", which_new=which_new)

            M_reconst = reshape(Array(reduce(*, M), reverse(sites)), 2^R)
            Mnew_reconst = reshape(Array(reduce(*, Mnew),
                                         vcat(reverse(sites), reverse(sites′))), 2^R, 2^R)

            @assert diag(Mnew_reconst) ≈ M_reconst
            @assert LinearAlgebra.diagm(M_reconst) ≈ Mnew_reconst
        end
    end

    @testset "shiftaxis" for R in [3], bc in [1, -1]
        sites = [Index(2, "Qubit, x=$n") for n in 1:R]
        g = randomMPS(sites)

        for shift in [0, 1, 2, 2^R-1]
            f = MSSTA.shiftaxis(g, shift, bc=bc)

            f_vec = vec(Array(reduce(*, f), reverse(sites)))
            g_vec = vec(Array(reduce(*, g), reverse(sites)))

            f_vec_ref = similar(f_vec)
            for i in 1:2^R
                ishifted = mod1(i + shift, 2^R)
                sign = ishifted == i + shift ? 1 : bc
                f_vec_ref[i] = g_vec[ishifted] * sign
            end

            @test f_vec_ref ≈ f_vec
        end
    end

    @testset "shiftaxis2d" for R in [3], bc in [1, -1]
        sitesx = [Index(2, "Qubit, x=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit, y=$n") for n in 1:R]
        sites = collect(Iterators.flatten(zip(sitesx, sitesy)))
        g = randomMPS(sites)

        for shift in [-4^R+1, -1, 0, 1, 2^R-1, 2^R, 2^R+1, 4^R+1]
            f = MSSTA.shiftaxis(g, shift, tag="x", bc=bc)

            f_mat = reshape(Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy))), 2^R, 2^R)
            g_mat = reshape(Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy))), 2^R, 2^R)

            f_mat_ref = similar(f_mat)
            for i in 1:2^R
                nbc, ishifted = divrem(i + shift - 1, 2^R, RoundDown)
                ishifted += 1
                f_mat_ref[i, :] = g_mat[ishifted, :] * (bc ^ nbc)
            end

            @test f_mat_ref ≈ f_mat
        end
    end
end

nothing

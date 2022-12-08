using Test
import MSSTA
using ITensors
using LinearAlgebra

@testset "util.jl" begin
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
end

nothing

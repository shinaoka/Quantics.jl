using Test
import MSSTA
using ITensors

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
end

nothing

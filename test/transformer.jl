using Test
import MSSTA
using ITensors

@testset "util.jl" begin @testset "flip" for nbit in 2:3,
                                             mostsignificantdigit in [:left, :right]

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
end end

nothing

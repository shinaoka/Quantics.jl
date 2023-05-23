using Test

using TensorCrossInterpolation
using MSSTA
using ITensors
ITensors.disable_warn_order()
using SparseIR: valueim, FermionicFreq

@testset "tci.jl" begin @testset "2D fermi gk" begin
    ek(kx, ky) = 2 * cos(kx) + 2 * cos(ky) - 1.0

    function gk(kx, ky, β)
        iv = valueim(FermionicFreq(1), β)
        1 / (iv - ek(kx, ky))
    end

    function f(xs, β)::ComplexF64
        kxy = 2π .* xs
        return gk(kxy[1], kxy[2], β)
    end

    R = 8
    N = 2^R
    halfN = 2^(R - 1)
    siteskx = [Index(2, "Qubit, kx=$n") for n in 1:R]
    sitesky = [Index(2, "Qubit, kx=$n") for n in 1:R]
    sitesk = [Index(4, "Quantics, k=$n") for n in 1:R]

    β = 10.0
    aqtt = MSSTA.construct_adaptiveqtt(ComplexF64, Val(2),
                                       x -> f(x, β), R; maxiter=50, tolerance=1e-5)

    M = MSSTA.asmps(aqtt, sitesk)

    truncate!(M; cutoff=1e-8)

    sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]
    sitesxy = collect(Iterators.flatten(zip(sitesx, sitesy)))

    M_qubit = MSSTA.splitsiteind(M, sitesxy; targetcsites=siteinds(M))
    truncate!(M_qubit; cutoff=1e-15)

    data = reshape(Array(reduce(*, M_qubit), reverse(sitesx)..., reverse(sitesy)...), 2^R,
                   2^R)

    xvec = collect(LinRange(0, 1, 2^R + 1)[1:(end - 1)])

    newaxis = [CartesianIndex()]
    f_(x, y) = f((x, y), β)
    data_ref = f_.(xvec[:, newaxis], xvec[newaxis, :])

    @test maximum(abs, data_ref .- data) < 1e-3 * maximum(abs, data_ref)
end end

nothing

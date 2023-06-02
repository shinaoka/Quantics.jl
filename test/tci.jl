using Distributed

using Test

# Define the maximum number of worker processes.
const MAX_WORKERS = 4

# Add worker processes if necessary.
addprocs(max(0, MAX_WORKERS - nworkers()))

@everywhere import MSSTA: QuanticsInd, originalcoordinate, DiscretizedGrid, adaptivetci
@everywhere using TensorCrossInterpolation
@everywhere import TensorCrossInterpolation as TCI
@everywhere using MSSTA
@everywhere using ITensors
@everywhere ITensors.disable_warn_order()
@everywhere using SparseIR: valueim, FermionicFreq

@testset "tci.jl" begin @testset "2D fermi gk" begin
    ek(kx, ky) = 2 * cos(kx) + 2 * cos(ky) - 1.0

    function gk(kx, ky, β)
        iv = valueim(FermionicFreq(1), β)
        1 / (iv - ek(kx, ky))
    end

    R = 20
    grid = MSSTA.DiscretizedGrid{2}(R, (0.0, 0.0), (2π, 2π))
    localdims = fill(4, R)

    β = 20.0
    f = x -> gk(originalcoordinate(grid, QuanticsInd{2}.(x))..., β)
    firstpivot = fill(4, R)
    firstpivot = TCI.optfirstpivot(f, localdims, firstpivot)
    absmaxval = abs(f(firstpivot))
    tol = 1e-5
    tree = MSSTA.adaptivetci(ComplexF64,
        f,
        localdims;
        maxbonddim=35, tolerance=tol)
    #@show tree

    for _ in 1:10
        pivot = rand(1:4, R)
        isapprox(
            MSSTA.evaluate(tree, pivot),
            f(pivot),
            atol=tol * absmaxval
        )
    end

    #==
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
    ==#
end end

nothing

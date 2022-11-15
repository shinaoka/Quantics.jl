using Test
using MultiScaleSpaceTimes
import ITensors: siteinds, Index
import ITensors
import SparseIR: Fermionic, Bosonic, FermionicFreq, valueim

function _test_data_imaginarytime(nbit, β)
    ω = 0.5
    N = 2^nbit
    halfN = 2^(nbit-1)

    # Tau
    gtau(τ) = - exp(-ω * τ) / (1 + exp(-ω * β))
    @assert gtau(0.0) + gtau(β) ≈ -1
    τs = collect(LinRange(0.0, β, N + 1))[1:end-1]
    gtau_smpl = Vector{ComplexF64}(gtau.(τs))

    # Matsubra
    giv(v::FermionicFreq) = 1/(valueim(v, β) - ω)
    vs = FermionicFreq.(2 .* collect(-halfN:halfN-1) .+ 1)
    giv_smpl = giv.(vs)

    return gtau_smpl, giv_smpl
end

@testset "imaginarytime.jl" begin
    @testset "decompose" begin
        β = 2.0
        nbit = 10
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sites = siteinds("Qubit", nbit)
        gtau_mps = MultiScaleSpaceTimes.decompose_gtau(gtau_smpl, sites; cutoff = 1e-20)

        gtau_smpl_reconst = vec(Array(reduce(*, gtau_mps), reverse(sites)...))

        @test gtau_smpl_reconst ≈ gtau_smpl
    end

    @testset "ImaginaryTimeFT.to_wn" begin
        ITensors.set_warn_order(100)
        β = 1.5
        nbit = 6
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sites = [Index(2, "Qubit,τ=$t,iω=$(nbit+1-t)") for t in 1:nbit]
        gtau_mps = MultiScaleSpaceTimes.decompose_gtau(gtau_smpl, sites; cutoff = 1e-20)
        ft = MultiScaleSpaceTimes.ImaginaryTimeFT(MultiScaleSpaceTimes.FTCore(sites))
        giv_mps = MultiScaleSpaceTimes.to_wn(Fermionic(), ft, gtau_mps, β; cutoff = 1e-20)

        # w_Q, ..., w_1
        giv =  vec(Array(reduce(*, giv_mps), sites...))

        #open("test.txt", "w") do file
            #for i in 1:nτ
                #println(file, i, " ", real(giv[i]), " ", imag(giv[i]), " ", real(giv_smpl[i]), " ", imag(giv_smpl[i]))
            #end
        #end

        @test maximum(abs, giv - giv_smpl) < 2e-2
        #@test false
    end

    @testset "ImaginaryTimeFT.to_tau" begin
        ITensors.set_warn_order(100)
        β = 1.5
        nbit = 8
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sites = [Index(2, "Qubit,τ=$t,iω=$(nbit+1-t)") for t in 1:nbit]
        giv_mps = MultiScaleSpaceTimes.decompose_giv(giv_smpl, sites; cutoff = 1e-20)

        ftcore = MultiScaleSpaceTimes.FTCore(sites)
        ft = MultiScaleSpaceTimes.ImaginaryTimeFT(ftcore)
        gtau_mps = MultiScaleSpaceTimes.to_tau(Fermionic(), ft, giv_mps, β; cutoff = 1e-20)

        # tau_Q, ..., tau_1
        #println("1 ", inds(t))
        gtau = vec(Array(reduce(*, gtau_mps), reverse(sites)...))

        #open("test.txt", "w") do file
            #for i in 1:nτ
                #println(file, i, " ", real(gtau[i]), " ", imag(gtau[i]), " ", real(gtau_smpl[i]), " ", imag(gtau_smpl[i]))
            #end
        #end

        # There is ocillation around tau = 0, beta.
        @test maximum(abs, (gtau - gtau_smpl)[trunc(Int,0.2*nτ):trunc(Int,0.8*nτ)]) < 1e-2
        #@test false
    end
end
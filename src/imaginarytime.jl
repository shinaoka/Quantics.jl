@doc """
For imaginary-time/-frequency domains
"""
struct ImaginaryTimeFT <: AbstractFT
    ftcore::FTCore

    function ImaginaryTimeFT(ftcore::FTCore)
        new(ftcore)
    end
end


function to_wn(::Fermionic, ft::ImaginaryTimeFT, gtau::MPS, beta::Float64; kwargs...)
    length(gtau) == nbit(ft) || error("Length mismatch")
    nbit_ = length(gtau)
    gtau = noprime(copy(gtau))

    N = 2^nbit_
    sites = extractsites(gtau)

    # Apply phase shift to each Qubit
    θ = π * ((-N+1)/N)
    for i in 1:nbit_
        gtau[i] = noprime(gtau[i] * op("Phase", sites[i]; ϕ= θ * 2^(nbit_-i) ))
    end

    # FFT
    M = forwardmpo(ft.ftcore, sites)
    giv = ITensors.apply(M, gtau; kwargs...)
    giv *= beta * 2^(-nbit_/2)

    return giv
end


function to_tau(::Fermionic, ft::ImaginaryTimeFT, giv::MPS, beta::Float64; kwargs...)
    length(giv) == nbit(ft) || error("Length mismatch")
    nbit_ = length(giv)
    giv = noprime(copy(giv))

    N = 2^nbit_
    sites = extractsites(giv)

    # Inverse FFT
    M = backwardmpo(ft.ftcore, sites; inputorder=:reversed)

    gtau = ITensors.apply(M, giv; kwargs...)

    gtau *= (2^(nbit_/2))/beta

    # Apply phase shift
    θ = - π * ((-N+1)/N)
    for i in 1:nbit_
        gtau[i] = noprime(gtau[i] * op("Phase", sites[i]; ϕ= θ * 2^(nbit_-i) ))
    end

    return gtau
end


function decompose_gtau(gtau_smpl::Vector{ComplexF64}, sites; kwargs...)
    nbit = length(sites)
    length(gtau_smpl) == 2^nbit || error("Length mismatch")

    # (g_1, g_2, ...)
    gtau_smpl = reshape(gtau_smpl, repeat([2,], nbit)...)
    gtau_smpl = permutedims(gtau_smpl, reverse(collect(1:nbit)))

    return MPS(gtau_smpl, sites; kwargs...)
end


function decompose_giv(giv_smpl::Vector{ComplexF64}, sites; kwargs...)
    nbit = length(sites)
    length(giv_smpl) == 2^nbit || error("Length mismatch")

    # (g_N, g_{N-1}, ...)
    giv_smpl = reshape(giv_smpl, repeat([2,], nbit)...)

    return MPS(giv_smpl, sites; kwargs...)
end


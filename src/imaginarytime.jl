
@doc """
For imaginary-time/-frequency domains
"""
struct ImaginaryTimeFT <: AbstractFT
    ftcore::FTCore

    function ImaginaryTimeFT(ftcore::FTCore)
        new(ftcore)
    end
end

_stat_shift(::Fermionic) = 1
_stat_shift(::Bosonic) = 0

_stat_sign(::Fermionic) = -1
_stat_sign(::Bosonic) = 1

function to_wn(stat::Statistics, gtau::MPS, beta::Float64; sitessrc=nothing, tag="",
               sitesdst=nothing, kwargs...)::MPS
    sitepos, _ = _find_target_sites(gtau; sitessrc=sitessrc, tag=tag)
    nqbit_t = length(sitepos)
    originwn = 0.5 * (-2.0^nqbit_t + _stat_shift(stat))
    giv = fouriertransform(gtau; tag=tag, sitessrc=sitessrc, sitesdst=sitesdst,
                           origindst=originwn)
    giv *= (beta * 2^(-nqbit_t / 2))
    return giv
end

function to_tau(stat::Statistics, giv::MPS, beta::Float64; sitessrc=nothing, tag="",
                sitesdst=nothing, kwargs...)::MPS
    sitepos, _ = _find_target_sites(giv; sitessrc=sitessrc, tag=tag)
    nqbit_t = length(sitepos)
    originwn = 0.5 * (-2.0^nqbit_t + _stat_shift(stat))
    gtau = fouriertransform(giv; sign=-1, tag=tag, sitessrc=sitessrc, sitesdst=sitesdst,
                            originsrc=originwn)
    gtau *= ((2^(nqbit_t / 2)) / beta)
    return gtau
end

function decompose_gtau(gtau_smpl::Vector{ComplexF64}, sites; kwargs...)
    nbit = length(sites)
    length(gtau_smpl) == 2^nbit || error("Length mismatch")

    # (g_1, g_2, ...)
    gtau_smpl = reshape(gtau_smpl, repeat([2], nbit)...)
    gtau_smpl = permutedims(gtau_smpl, reverse(collect(1:nbit)))

    return MPS(gtau_smpl, sites; kwargs...)
end

"""
w = (w_1 w_2, ..., w_R)_2
In the resultant MPS, the site indices are
w_R, w_{R-1}, ..., w_1 from the left to the right.

sites: indices for w_1, ..., w_R in this order.
"""
function decompose_giv(giv_smpl::Vector{ComplexF64}, sites; kwargs...)
    nbit = length(sites)
    length(giv_smpl) == 2^nbit || error("Length mismatch")
    tensor = ITensor(giv_smpl, reverse(sites))
    return MPS(tensor, reverse(sites); kwargs...)
end

"""
Construct an MPS representing G(τ) generated by a pole
"""
function poletomps(stat::Statistics, sites, β, ω)
    nqubits = length(sites)
    links = [Index(1, "Link,l=$l") for l in 0:nqubits]
    tensors = ITensor[]
    for n in 1:nqubits
        push!(tensors,
              ITensor([1.0, exp(-(0.5^n) * β * ω)], links[n], links[n + 1], sites[n]))
    end
    tensors[1] *= -1 / (1 - _stat_sign(stat) * exp(-β * ω))
    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)
    return MPS(tensors)
end

poletomps(sites, β, ω) = poletomps(Fermionic(), sites, β, ω)

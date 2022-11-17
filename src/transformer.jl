abstract type AbstractFT end

struct FTCore
    forward::MPO

    function FTCore(sites; kwargs...)
        new(_qft(sites; kwargs...))
    end
end

nbit(ft::AbstractFT) = length(ft.ftcore.forward)


"""
sites[1] corresponds to the most significant digit.
sign = 1
"""
function forwardmpo(ftcore::FTCore, sites)
    M = copy(ftcore.forward)
    replace_mpo_siteinds!(M, extractsites(M), sites)
    return M
end


function backwardmpo(ftcore::FTCore, sites)
    M = conj(MPO(reverse([x for x in ftcore.forward])))
    replace_mpo_siteinds!(M, extractsites(M), sites)
    return M
end


@doc raw"""

sitesrc[1] and sitesrc[end] correspond to the most significant and least significant
digits of the input, respectively.

sitedst[1] and sitedst[end] correspond to the most significant and least significant
digits of the output, respectively.

"""
function qft(
        M::MPS;
        sign::Int=1,
        tag::String="",
        sitessrc = nothing,
        sitesdst = nothing,
        cutoff_MPO=1e-25, kwargs...)
    if tag == "" && sitessrc === nothing
        error("tag or sitesrc must be specified")
    end

    # Set input site indices
    if tag != ""
        sites = siteinds(M)
        sitepos = findallsites_by_tag(sites; tag=tag)
        target_sites = [sites[p] for p in sitepos]
    elseif sitesrc !== nothing
        target_sites = sitesrc
        sitepos = [findsite(M, s) for s in target_sites]
    end

    if sitesdst === nothing
        sitesdst = target_sites
    end

    MQ_ = _qft(target_sites; sign=sign, cutoff=cutoff_MPO)
    MQ = matchsiteinds(MQ_, sites)
    M_result = apply(MQ, M; kwargs...)

    N = length(target_sites)
    for n in eachindex(target_sites)
        replaceind!(M_result[sitepos[n]], target_sites[n], sitesdst[N-n+1])
    end

    return M_result
end
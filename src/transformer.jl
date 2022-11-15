abstract type AbstractFT end

struct FTCore
    forward::MPO

    function FTCore(sites; kwargs...)
        new(_qft(sites; kwargs...))
    end
end

nbit(ft::AbstractFT) = length(ft.ftcore.forward)


"""

inputorder:
    If inputorder==:normal, sites[1] corresponds to the most significant digit.
    If inputorder==:reversed, sites[1] corresponds to the least significant digit.
"""
function forwardmpo(ftcore::FTCore, sites; inputorder=:normal)
    inputorder ∈ [:normal, :reversed] || error("Invalid inputorder")

    if inputorder == :normal
        M = copy(ftcore.forward)
        replace_mpo_siteinds!(M, extractsites(M), sites)
        return M
    else
        error("Not implemented yet")
    end
end


function backwardmpo(ftcore::FTCore, sites; inputorder=:normal)
    inputorder ∈ [:normal, :reversed] || error("Invalid inputorder")

    if inputorder == :normal
        error("Not implemented yet")
    else
        M = conj(MPO(reverse([x for x in ftcore.forward])))
        replace_mpo_siteinds!(M, extractsites(M), sites)
        return M
    end
end
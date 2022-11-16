"""
Match the site indices of the MPO and the given site indices
"""
function matchsiteinds!(M::MPO, sites) where {T}
    if all(siteinds(M) .== sites)
        return nothing
    end
end


# MPO with edges
mutable struct MPOEdge <: AbstractMPSEdge
    data::Vector{ITensor}
    siteinds::Vector{Vector{Index{Int}}}
    linkinds::Vector{Index{Int}}
end

function addedges(M::MPO)::MPOEdge
    linkinds_org = linkinds(M)
    siteinds_org = siteinds(M)
    # data may be copied?
    M = copy(M)
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(M))")
    M[1] = ITensor(ITensors.data(M[1]), [linkl, inds(M[1])...])
    M[end] = ITensor(ITensors.data(M[end]), [inds(M[end])..., linkr])
    return MPOEdge([x for x in M], siteinds_org, vcat(linkl, linkinds_org, linkr))
end

#ITensors.siteinds(M::MPOEdge) = M.siteinds
#ITensors.linkinds(M::MPOEdge) = M.linkinds
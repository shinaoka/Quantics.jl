abstract type AbstractMPSEdge <: ITensors.AbstractMPS end

# MPS with edges
mutable struct MPSEdge <: AbstractMPSEdge
    data::Vector{ITensor}
    siteinds::Vector{Index{Int}}
    linkinds::Vector{Index{Int}}
end

ITensors.siteinds(M::AbstractMPSEdge) = M.siteinds
ITensors.linkinds(M::AbstractMPSEdge) = M.linkinds

function addedges(M::Union{MPS,MPO})
    linkinds_org = linkinds(M)
    siteinds_org = siteinds(M)
    # data may be copied?
    M = copy(M)
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(M))")
    M[1] = ITensor(ITensors.data(M[1]), [linkl, inds(M[1])...])
    M[end] = ITensor(ITensors.data(M[end]), [inds(M[end])..., linkr])
    T = (typeof(M) == MPS ? MPSEdge : MPOEdge)
    links = [linkl, linkinds_org..., linkr]
    tensors = [x for x in M]
    return T(tensors, siteinds_org, links)
end


# MPO with edges
mutable struct MPOEdge <: AbstractMPSEdge
    data::Vector{ITensor}
    siteinds::Vector{Vector{Index{Int}}}
    linkinds::Vector{Index{Int}}
end
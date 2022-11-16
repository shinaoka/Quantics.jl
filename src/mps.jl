"""
Create a MPS filled with one
"""
function onemps(::Type{T}, sites) where {T<:Number}
    M = MPS(T, sites; linkdims=1)
    l = linkinds(M)
    for n in eachindex(M)
        if n == 1
            M[n] = ITensor(T, sites[n], l[n])
        elseif n == length(M)
            M[n] = ITensor(T, l[n-1], sites[n])
        else
            M[n] = ITensor(T, l[n-1], sites[n], l[n])
        end
        M[n] .= one(T)
    end
    return M
end

abstract type AbstractMPSEdge <: ITensors.AbstractMPS end

# MPS with edges
mutable struct MPSEdge <: AbstractMPSEdge
    data::Vector{ITensor}
    siteinds::Vector{Index{Int}}
    linkinds::Vector{Index{Int}}
end

ITensors.siteinds(M::AbstractMPSEdge) = M.siteinds
ITensors.linkinds(M::AbstractMPSEdge) = M.linkinds

function addedges(M::MPS)::MPSEdge
    linkinds_org = linkinds(M)
    siteinds_org = siteinds(M)
    # data may be copied?
    M = copy(M)
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(M))")
    M[1] = ITensor(ITensors.data(M[1]), [linkl, inds(M[1])...])
    M[end] = ITensor(ITensors.data(M[end]), [inds(M[end])..., linkr])
    return MPSEdge([x for x in M], siteinds_org, vcat(linkl, linkinds_org, linkr))
end
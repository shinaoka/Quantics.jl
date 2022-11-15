"""
MPS

A finite size matrix product state type.
A site index of an MPS can be a list of indices.
"""
mutable struct _MPS
    data::Vector{ITensor}
    siteinds::Vector{Vector{Index{Int}}}
end


"""
MPO

A finite size matrix product operator type.
A site index of an MPS can be a list of indices.
"""
mutable struct _MPO
    data::Vector{ITensor}
    siteinds::Vector{Vector{Index{Int}}}
end
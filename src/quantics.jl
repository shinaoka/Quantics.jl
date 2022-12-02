"""
Quantics index (1-based, ≤2^N)
"""
struct QuanticsInd{N}
    data::Int
    function QuanticsInd{N}(i) where {N}
        1 ≤ i ≤ 2^N || error("Invalid i for QuanticsInd{$N}: $i")
        return new{N}(i)
    end
end

Base.convert(::Type{Int}, x::QuanticsInd{N}) where {N} = x.data

const QubitInd = QuanticsInd{2}

Base.convert(::Type{QubitInd}, x::T) where {T<:Integer} = QubitInd(x)

function QuanticsInd{N}(inds::NTuple{N,QubitInd}) where {N}
    i = 0
    c = 2^(N - 1)
    for n in 1:N
        i += c * (convert(Int, inds[n]) - 1)
        c = c >> 1
    end
    return QuanticsInd{N}(i + 1)
end

function asqubits(inds::NTuple{N,Int}) where {N}
    return Tuple(QubitInd.(inds))
end

"""
Convert a 1-based quantics integer index to 1-based qubit indices
"""
function asqubit(idx::QuanticsInd{N}) where {N}
    i = convert(Int, idx)
    b = zeros(MVector{N,Int})
    i -= 1
    for n in 1:N
        i, b[N - n + 1] = divrem(i, 2)
    end
    b .+= 1
    return NTuple{N,QubitInd}(b)
end

"""
Convert 1-based quantics indices to 1-based qubit indices
"""
function asqubits(indices::AbstractVector{QuanticsInd{N}})::Vector{QubitInd} where {N}
    return collect(Iterators.flatten(asqubit.(indices)))
end

"""
Convert 1-based qubit indices to 1-based quantics indices
"""
function asquantics(::Val{N},
                    indices::AbstractVector{QubitInd})::Vector{QuanticsInd{N}} where {N}
    indices_ = reshape(indices, N, :)
    nquantics = size(indices_, 2)
    return collect(QuanticsInd{N}(Tuple(indices_[:, n])) for n in 1:nquantics)
end

function quantics_to_index(inds::AbstractVector{QuanticsInd{N}}) where {N}
    nquantics = length(inds)
    qubits = reshape(asqubits(inds), N, nquantics)

    res = ones(MVector{N,Int})
    c = 2^(nquantics - 1)
    for iq in 1:nquantics
        for n in 1:N
            res[n] += c * (convert(Int, qubits[n, iq]) - 1)
        end
        c = c >> 1
    end
    return NTuple{N,Int}(res)
end

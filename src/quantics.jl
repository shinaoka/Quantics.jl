"""
Convert a 1-based quantics integer index to 1-based qubit indices
"""
function quantics_to_qubit(::Val{N}, i::Int) where {N}
    @assert 1 ≤ i ≤ 2^N
    b = zeros(MVector{N,Int})
    i -= 1
    for n in 1:N
        i, b[N - n + 1] = divrem(i, 2)
    end
    return b .+ 1
end

"""
Convert 1-based quantics indices to 1-based qubit indices
"""
function quantics_to_qubit(::Val{N}, indices::AbstractVector{Int}) where {N}
    return collect(Iterators.flatten(quantics_to_qubit.(Val(N), indices)))
end

"""
Convert 1-based qubit indicse to a 1-based quantics index
"""
function qubit_to_quantics(indices)
    all(1 .≤ indices .≤ 2) || error("Invalid indices")
    N = length(indices)
    i = 0
    c = 2^(N - 1)
    for n in eachindex(indices)
        i += c * (indices[n] - 1)
        c = c >> 1
    end
    return i + 1
end

"""
Convert 1-based qubit indices to 1-based quantics indices
"""
function qubit_to_quantics(::Val{N}, indices::Vector{Int}) where {N}
    indices_ = reshape(indices, N, :)
    nquantics = size(indices_, 2)
    return collect(Iterators.flatten(qubit_to_quantics(indices_[:, n]) for n in 1:nquantics))
end


"""
Convert 1-based quantics indices of size 2^N to N integer indices
"""
function quantics_to_index(::Val{N}, quantics_inds::Vector{Int}) where {N}
    nquantics = length(quantics_inds)
    qubit_inds = reshape(quantics_to_qubit(Val(N), quantics_inds), N, nquantics)

    res = ones(MVector{N,Int})
    c = 2^(nquantics-1)
    for iq in 1:nquantics
        for n in 1:N
            res[n] += c * (qubit_inds[n, iq] - 1)
        end
        c = c >> 1
    end
    return NTuple{N,Int}(res)
end


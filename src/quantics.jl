"""
Convert a 1-based quantics integer index to 1-based qubit indices
"""
function quantics_to_qubit(::Val{N}, i::Int) where {N}
    @assert 1 ≤ i ≤ 2^N
    b = Vector{Int}(undef, N)
    i -= 1
    for n in 1:N
        i, b[N - n + 1] = divrem(i, 2)
    end
    return NTuple{N,Int}(b .+ 1)
end

"""
Convert 1-based quantics indices to 1-based qubit indices
"""
function quantics_to_qubit(::Val{N}, indices::Vector{Int}) where {N}
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

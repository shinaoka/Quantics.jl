"""
Quantics index (1-based, ≤2^D) for expanding a D-dimensional space
"""
struct QuanticsInd{D}
    data::Int
    function QuanticsInd{D}(i) where {D}
        1 ≤ i ≤ 2^D || error("Invalid i for QuanticsInd{$D}: $i")
        return new{D}(i)
    end
end

Base.convert(::Type{Int}, x::QuanticsInd{D}) where {D} = x.data
const QubitInd = QuanticsInd{1}
quanticsdim(::QuanticsInd{D}) where {D} = D

Base.convert(::Type{QubitInd}, x::T) where {T<:Integer} = QubitInd(x)

"""
D QubitInd objects are packed into a single QuanticsInd{D} object,
which represent a single length-scale layer of the `D`-dimensional space.
"""
function QuanticsInd{D}(inds::NTuple{D,QubitInd}) where {D}
    i = 0
    c = 2^(D - 1)
    for n in 1:D
        i += c * (convert(Int, inds[n]) - 1)
        c = c >> 1
    end
    return QuanticsInd{D}(i + 1)
end

# Deprecated?
function asqubits(inds::NTuple{D,Int}) where {D}
    return Tuple(QubitInd.(inds))
end

"""
Convert an integer index (1 <= index <= 2^R) into R qbits
"""
function asqubits(index::Int, R::Int)
    1 <= index <= 2^R || error("index is out of range")

    bits = Vector{Int}(undef, R)
    index_ = index - 1

    for i in 1:R
        bits[R-i+1] = mod(index_, 2) + 1
        index_ = index_ ÷ 2
    end

    return QubitInd.(bits)
end


"""
Convert a 1-based quantics integer index to 1-based qubit indices
"""
function asqubit(idx::QuanticsInd{D}) where {D}
    i = convert(Int, idx)
    b = zeros(MVector{D,Int})
    i -= 1
    for n in 1:D
        i, b[D - n + 1] = divrem(i, 2)
    end
    b .+= 1
    return NTuple{D,QubitInd}(b)
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
function asquantics(::Val{D},
                    indices::AbstractVector{QubitInd})::Vector{QuanticsInd{D}} where {D}
    indices_ = reshape(indices, D, :)
    nquantics = size(indices_, 2)
    return collect(QuanticsInd{D}(Tuple(indices_[:, n])) for n in 1:nquantics)
end

function quantics_to_index(inds::AbstractVector{QuanticsInd{D}}) where {D}
    nquantics = length(inds)
    qubits = reshape(asqubits(inds), D, nquantics)

    res = ones(MVector{D,Int})
    c = 2^(nquantics - 1)
    for iq in 1:nquantics
        for n in 1:D
            res[n] += c * (convert(Int, qubits[n, iq]) - 1)
        end
        c = c >> 1
    end
    return NTuple{D,Int}(res)
end

#==
function quantics_to_index(index::NTuple{D,Int}, R::Int) where {D}
    for i in index
        1 <= i <= 2^R || error("index is out of range!")
    end

    #qubits = reshape(asqubits(inds), N, nquantics)

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
==#

"""
indices:
    (x_1, x_2, ..., x_D), where 1 <= x_i <= 2^R. Reprenents a point in D-dimensional space.

Return: 
    A QuanticsInd{D} string reprensenting the point.
"""
function index_to_quantics(indices::NTuple{D,Int}, R::Int) where {D}
    qubit_strings = asqubits.(indices, R)
    return asquantics(Val(D), collect(Iterators.flatten(zip(qubit_strings...))))
end


"""
Convert qubit string for one-dimensional space into a integer index
"""
qubit_to_index(inds::AbstractVector{QubitInd})::Int = MSSTA.quantics_to_index(inds)[1]


"""
Convert qubit string for D-dimensional space into integer indices
"""
function qubit_to_index(::Val{D}, inds::AbstractVector{QubitInd}, R::Int) where D
    D * R == length(inds) || error("Length of inds does not match R*D")

    return NTuple{D,Int}(qubit_to_index(inds[i:D:end]) for i in 1:D)
end

"""
Convert integer indices representing a point in D-dimensional space into qubit string
"""
function index_to_qubit(indices::NTuple{D,Int}, R::Int) where D
    return MSSTA.asqubits(MSSTA.index_to_quantics(indices, R))
end
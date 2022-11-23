"""
For elementwise/matrix multiplication
"""
struct Multiplier
end

abstract type AbstractMultiplier end

struct MatrixMultiplier{T} <: AbstractMultiplier where {T<:Number}
    #sites_row::Vector{Index{T}}
    #sites_shared::Vector{Index{T}}
    #sites_col::Vector{Index{T}}
    tag_row::String
    tag_shared::String
    tag_col::String

    function MatrixMultiplier(tag_row::String, tag_shared::String, tag_col::String)
        #function _preprocess_matmul!(tensors1::Vector{ITensor},
            #tensors2::Vector{ITensor},
            #sites1,
            #sites2,
            #tag_row::String, tag_shared::String, tag_col::String)
        length(unique([tag_row, tag_shared, tag_col])) == 3 ||
           error("tag_row, tag_shared, tag_col must be different")
        new(tag_row, tag_shared, tag_col)
    end
end


"""
Contract two MPS tensors to form an MPO

sites1[i] and sites2[i] must be at two adjacent sites.
Each pair of the two adjacent tensors is contracted.
"""
function _preprocess_matmul!(tensors::Vector{ITensor}, sites1::Vector{Index{T}},
                             sites2::Vector{Index{T}}) where {T}
    for (s1, s2) in zip(sites1, sites2)
        p1 = findfirst(x -> hasind(x, s1), tensors)
        p2 = findfirst(x -> hasind(x, s2), tensors)
        p1 === nothing && error("Not found $s1")
        p2 === nothing && error("Not found $s2")
        abs(p1 - p2) == 1 ||
            error("$s1 and $s2 are found at indices $p1 and $p2. They must be on two adjacent sites.")
        idx = min(p1, p2)
        tensor = tensors[idx] * tensors[idx + 1]
        deleteat!(tensors, idx:(idx + 1))
        insert!(tensors, idx, tensor)
    end

    return nothing
end

"""
tag_row and tag_shared for tensors1
tag_col and tag_shared for tensors2
"""
function _preprocess_matmul!(tensors1::Vector{ITensor},
                             tensors2::Vector{ITensor},
                             sites1,
                             sites2,
                             tag_row::String, tag_shared::String, tag_col::String)
    sites_row = findallsiteinds_by_tag(sites1; tag=tag_row)
    sites_shared = findallsiteinds_by_tag(sites1; tag=tag_shared)
    sites_col = findallsiteinds_by_tag(sites2; tag=tag_col)

    # TODO: more checkes on the order of sites in sites_row, ...

    _preprocess_matmul!(tensors1, sites_row, sites_shared)
    _preprocess_matmul!(tensors2, sites_col, sites_shared)

    return nothing
end


function _postprocess_matmul(
    M::MPO, sites_row::Vector{Index{T}}, sites_col::Vector{Index{T}})::MPO where T
    for (s1, s2) in zip(sites_row, sites_col)
        p = findsite(M, s1)
        hasind(M[p], s2) || error("$s1 and $s2 are not on the same site")

        indsl = [s1]
        if p > 1
            push!(indsl, linkind(M, p-1))
        end

        indsr = [s2]
        if p < length(M)
            push!(indsr, linkind(M, p))
        end

        Ml, Mr = split_tensor(M[p], [indsl, indsr])

        tensors = ITensors.data(M)
        deleteat!(tensors, p)
        insert!(tensors, p, Ml)
        insert!(tensors, p+1, Mr)

        M = _convert_to_MPO(tensors)
    end

    return M
end

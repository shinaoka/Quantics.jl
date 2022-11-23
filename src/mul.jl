"""
For elementwise/matrix multiplication
"""
struct Multiplier
end

function preprosses(mul::Multiplier, M1::MPS, M2::MPS)
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

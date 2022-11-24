abstract type AbstractMultiplier end

#===
Matrix multiplication
===#
struct MatrixMultiplier{T} <: AbstractMultiplier where {T}
    site_row::Index{T}
    site_shared::Index{T}
    site_col::Index{T}

    function MatrixMultiplier(site_row::Index{T},
                              site_shared::Index{T},
                              site_col::Index{T}) where {T}
        new{T}(site_row, site_shared, site_col)
    end
end

function MatrixMultipliers(sites_row::Vector{Index{T}},
                           sites_shared::Vector{Index{T}},
                           sites_col::Vector{Index{T}}) where {T}
    return [MatrixMultiplier(s...) for s in zip(sites_row, sites_shared, sites_col)]
end

function preprocess(mul::MatrixMultiplier{T}, M1::MPO, M2::MPO) where {T}
    return combinesites(M1, mul.site_row, mul.site_shared),
           combinesites(M2, mul.site_col, mul.site_shared)
end

function postprocess(mul::MatrixMultiplier{T}, M::MPO)::MPO where {T}
    site_row = mul.site_row
    site_col = mul.site_col

    p = findsite(M, site_row)
    hasind(M[p], site_col) || error("$site_row and $site_col are not on the same site")

    indsl = [site_row]
    if p > 1
        push!(indsl, linkind(M, p - 1))
    end

    indsr = [site_col]
    if p < length(M)
        push!(indsr, linkind(M, p))
    end

    Ml, Mr = split_tensor(M[p], [indsl, indsr])

    tensors = ITensors.data(M)
    deleteat!(tensors, p)
    insert!(tensors, p, Ml)
    insert!(tensors, p + 1, Mr)

    return MPO(tensors)
end

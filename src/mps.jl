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
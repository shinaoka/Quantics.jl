#function _siteinds(d::Vector{T}; kwargs...) where {T<:Integer}
    #return [siteind(d[n], n; kwargs...) for n in eachindex(d)]
#end

function extractsite(x::Union{MPS,MPO}, n::Int)
    if n == 1
        return noprime(copy(uniqueind(x[n], x[n+1])))
    elseif n == length(x)
        return noprime(copy(uniqueind(x[n], x[n-1])))
    else
        return noprime(copy(uniqueind(x[n], x[n+1], x[n-1])))
    end
end

extractsites(x::Union{MPS,MPO}) = [extractsite(x, n) for n in eachindex(x)]


function replace_mpo_siteinds!(M::MPO, sites_src, sites_dst)
    sites_src = noprime(sites_src)
    sites_dst = noprime(sites_dst)
    for j in eachindex(M)
        replaceind!(M[j], sites_src[j], sites_dst[j])
        replaceind!(M[j], sites_src[j]', sites_dst[j]')
    end
    return M
end



"""
Reverse the order of the physical indices of a MPO
"""
#revserMPO(reverse([x for x in M]))


"""
Create a MPO with ITensor objects of ElType ComplexF64 filled with zero
"""
function _zero_mpo(sites; linkdims=ones(Int, length(sites)-1))
    length(linkdims) == length(sites) - 1 || error("Length mismatch $(length(linkdims)) != $(length(sites)) - 1")
    M = MPO(sites)
    N = length(M)
    links = [Index(1, "n=0,Link")]
    for n in 1:(N-1)
        push!(links, Index(linkdims[n], "n=$(n),Link"))
    end
    push!(links, Index(1, "n=$N,Link"))
    for n in 1:N
        inds_ = (links[n], sites[n]', sites[n], links[n + 1])
        elm_ = zeros(ComplexF64, map(ITensors.dim, inds_)...)
        M[n] = ITensor(elm_, inds_...)
    end
    M[1] *= ITensors.delta(links[1])
    M[N] *= ITensors.delta(links[N + 1])
    
    return M
end


# Compute linkdims for a maximally entangled state
function maxlinkdims(inds)
    N = length(inds)
    for i in 1:N
        @assert !ITensors.hastags(inds, "Link")
    end

    physdims = dim.(inds)

    maxdim = ones(Float64, N-1)
    maxdiml = 1.0
    for i in 1:(N-1)
        maxdiml *= physdims[i]
        maxdim[i] = maxdiml
    end

    maxdimr = 1.0
    for i in 1:(N-1)
        maxdimr *= physdims[N+1-i]
        maxdim[N-i] = min(maxdimr, maxdim[N-i])
    end
    return maxdim
end

linkdims(M) = [dim(ITensors.commonind(M[n], M[n+1])) for n in 1:(length(M)-1)]

links(M) = [ITensors.commonind(M[n], M[n+1]) for n in 1:(length(M)-1)]

function _split(t::ITensor, csite, outerlinks, sites)
    length(sites) == 2 || error("Length of sites must be 2")

    Dleft = dim(outerlinks[1])
    Dright = dim(outerlinks[2])
    #prod(size(t)) == 4 * Dleft * Dright || @show t
    #prod(size(t)) == 4 * Dleft * Dright || @show csite
    #prod(size(t)) == 4 * Dleft * Dright || @show outerlinks
    #prod(size(t)) == 4 * Dleft * Dright || @show sites
    prod(size(t)) == 4 * Dleft * Dright || error("Length mismatch")
    t = permute(t, [outerlinks[1], csite, outerlinks[2]])
    sites_ = [outerlinks[1], sites..., outerlinks[2]]
    t = ITensor(ITensors.data(t), sites_...)
    U, S, V = svd(t, sites_[1], sites_[2])
    SV = S * V
    return U, SV
end


function _splitsiteind(M::MPS, sites, s1, s2, csite)
    hasedge(M) || error("M must have edges")

    n = findsite(M, csite) 
    l = _linkinds(M, sites)
    tensors = _split(M[n], csite, l[n:n+1], [s1, s2])
    return MPS([M[1:n-1]..., tensors..., M[n+1:end]...]),
        [sites[1:n-1]..., s1, s2, sites[n+1:end]...]
end


function splitsiteind(M::MPS, sites; targetcsites=siteinds(M))
    !hasedge(M) || error("M must not have edges")
    2 * length(targetcsites) == length(sites) || error("Length mismatch")

    sites_M = siteinds(M)
    sites_res = siteinds(M)
    addedges!(M) # This will be canceled out by the following removeedges!

    res = copy(M)
    for n in eachindex(targetcsites)
        res, sites_res = _splitsiteind(res, sites_res, sites[2*n-1], sites[2*n], targetcsites[n])
    end

    removeedges!(M, sites_M)
    removeedges!(res, sites_res)
    return res
end

splitsiteinds = splitsiteind

function addedges!(x::MPS)
    length(inds(x[1])) == 2 || error("Dim of the first tensor must be 2")
    length(inds(x[end])) == 2 || error("Dim of the last tensor must be 2")
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(x))")
    x[1] = ITensor(ITensors.data(x[1]), [linkl, inds(x[1])...])
    x[end] = ITensor(ITensors.data(x[end]), [inds(x[end])..., linkr])
    return nothing
end


function addedges!(x::MPO)
    length(inds(x[1])) == 3 || error("Dim of the first tensor must be 3")
    length(inds(x[end])) == 3 || error("Dim of the last tensor must be 3")
    linkl = Index(1, "Link,l=0")
    linkr = Index(1, "Link,l=$(length(x))")
    x[1] = ITensor(ITensors.data(x[1]), [linkl, inds(x[1])...])
    x[end] = ITensor(ITensors.data(x[end]), [inds(x[end])..., linkr])
    return nothing
end


function removeedges!(x::MPS, sites)
    length(inds(x[1])) == 3 || error("Dim of the first tensor must be 3")
    length(inds(x[end])) == 3 || error("Dim of the last tensor must be 3")
    elt = eltype(x[1])
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites)=>1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end-1], sites)=>1)
    return nothing
end


function removeedges!(x::MPO, sites)
    length(inds(x[1])) == 4 || error("Dim of the first tensor must be 4")
    length(inds(x[end])) == 4 || error("Dim of the last tensor must be 4")
    elt = eltype(x[1])
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites, prime.(sites))=>1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end-1], sites, prime.(sites))=>1)
    return nothing
end


function _combinesiteinds(t1::ITensor, t2::ITensor, s1, s2, csite)
    t = t1 * t2
    if dim(t1) == 2
        # Left edge
        l = uniqueinds(inds(t), [s1, s2])
        return ITensor(Array(t, [s1, s2, l]), [csite, l])
    elseif dim(t2) == 2
        # right edge
        l = uniqueinds(inds(t), [s1, s2])
        return ITensor(Array(t, [l, s1, s2]), [l, csite])
    else
        l1 = uniqueinds(inds(t), s1, s2, t2)
        l2 = uniqueinds(inds(t), s1, s2, t1)
        return ITensor(Array(t, [l1, s1, s2, l2]), [l1, csite, l2])
    end
end


function _combinesiteinds(M::MPS, tsite1, tsite2, csite)
    !hasedge(M) || error("MPS must not have edges")

    sp = findsite(M, tsite1)
    siteind(M, sp+1) == tsite2 || error("Found wrong site")

    ctensor = _combinesiteinds(M[sp], M[sp+1], tsite1, tsite2, csite)
    return MPS([M[1:sp-1]..., ctensor, M[sp+2:end]...])
end


function combinesiteinds(M::MPS, csites; targetsites::Vector=siteinds(M))
    !hasedge(M) || error("MPS must not have edges")
    length(targetsites) == 2*length(csites) || error("Length mismatch")
    for n in eachindex(csites)
        M = _combinesiteinds(M, targetsites[2*n-1], targetsites[2*n], csites[n])
    end
    return M
end


_mklinks(dims) = [Index(dims[l], "Link,l=$l") for l in eachindex(dims)]


hasedge(M::MPS) = (length(inds(M[1])) == 3)


function _linkinds(M::MPS, sites::Vector{T}) where T
    N = length(M)
    if hasedge(M)
        links = T[]
        push!(links, uniqueind(M[1], M[2], sites))
        for n in 1:(N-1)
            push!(links, commonind(M[n], M[n+1]))
        end
        push!(links, uniqueind(M[end], M[end-1], sites))
        return links
    else
        return linkinds(M)
    end
end


"""
Decompose a tensor into a set of indices by QR
"""
function split_tensor(tensor::ITensor, inds_list::Vector{Vector{Index{T}}}) where T
    result = ITensor[]
    for (i, inds) in enumerate(inds_list)
        if i == length(inds_list)
            push!(result, tensor)
        else
            Q, R, _ = qr(tensor, inds)
            push!(result, Q)
            tensor = R
        end
    end
    return result
end


function cleanup_linkinds!(M)
    links_new = [Index(dim(l), "Link,l=$idx") for (idx, l) in enumerate(linkinds(M))]
    links_old = linkinds(M)
    for n in 1:length(M)
        if n < length(M)
            replaceind!(M[n], links_old[n], links_new[n])
        end
        if n > 1
            replaceind!(M[n], links_old[n-1], links_new[n-1])
        end
    end
end

"""
To digits
"""
function tobin!(x::Int, xbin::Vector{Int})
    nbit = length(xbin)
    mask = 1 << (nbit-1)
    for i in 1:nbit
        xbin[i] = (mask & x) >> (nbit - i)
        mask = mask >> 1
    end
end

# Get bit at pos (>=0). pos=0 is the least significant digit.
_getbit(i, pos) = ((i & (1 << pos)) >> pos)

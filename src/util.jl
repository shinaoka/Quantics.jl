#function _siteinds(d::Vector{T}; kwargs...) where {T<:Integer}
#return [siteind(d[n], n; kwargs...) for n in eachindex(d)]
#end

function extractsite(x::Union{MPS,MPO}, n::Int)
    if n == 1
        return noprime(copy(uniqueind(x[n], x[n + 1])))
    elseif n == length(x)
        return noprime(copy(uniqueind(x[n], x[n - 1])))
    else
        return noprime(copy(uniqueind(x[n], x[n + 1], x[n - 1])))
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
Reverse the order of the physical indices of an MPS/MPO
"""
#function _reverse(M::MPS) = typeof(M)([M[n] for n in reverse(1:length(M))])
function _reverse(M::MPO)
    sites = extractsites(M)
    N = length(M)
    M_ = MPO([M[n] for n in reverse(1:length(M))])
    for n in 1:N
        replaceind!(M_[n], sites[N - n + 1], sites[n])
        replaceind!(M_[n], sites[N - n + 1]', sites[n]')
    end
    return M_
end

"""
Create a MPO with ITensor objects of ElType ComplexF64 filled with zero
"""
function _zero_mpo(sites; linkdims=ones(Int, length(sites) - 1))
    length(linkdims) == length(sites) - 1 ||
        error("Length mismatch $(length(linkdims)) != $(length(sites)) - 1")
    M = MPO(sites)
    N = length(M)
    links = [Index(1, "n=0,Link")]
    for n in 1:(N - 1)
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

    maxdim = ones(Float64, N - 1)
    maxdiml = 1.0
    for i in 1:(N - 1)
        maxdiml *= physdims[i]
        maxdim[i] = maxdiml
    end

    maxdimr = 1.0
    for i in 1:(N - 1)
        maxdimr *= physdims[N + 1 - i]
        maxdim[N - i] = min(maxdimr, maxdim[N - i])
    end
    return maxdim
end

linkdims(M) = [dim(ITensors.commonind(M[n], M[n + 1])) for n in 1:(length(M) - 1)]

links(M) = [ITensors.commonind(M[n], M[n + 1]) for n in 1:(length(M) - 1)]

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
    tensors = _split(M[n], csite, l[n:(n + 1)], [s1, s2])
    return MPS([M[1:(n - 1)]..., tensors..., M[(n + 1):end]...]),
           [sites[1:(n - 1)]..., s1, s2, sites[(n + 1):end]...]
end

function splitsiteind(M::MPS, sites; targetcsites=siteinds(M))
    !hasedge(M) || error("M must not have edges")
    2 * length(targetcsites) == length(sites) || error("Length mismatch")

    sites_res = siteinds(M)
    M = deepcopy(M)
    addedges!(M)

    res = copy(M)
    for n in eachindex(targetcsites)
        res, sites_res = _splitsiteind(res, sites_res, sites[2 * n - 1], sites[2 * n],
                                       targetcsites[n])
    end

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
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites) => 1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end - 1], sites) => 1)
    return nothing
end

function removeedges!(x::MPO, sites)
    length(inds(x[1])) == 4 || error("Dim of the first tensor must be 4")
    length(inds(x[end])) == 4 || error("Dim of the last tensor must be 4")
    elt = eltype(x[1])
    x[1] *= onehot(elt, uniqueind(x[1], x[2], sites, prime.(sites)) => 1)
    x[end] *= onehot(elt, uniqueind(x[end], x[end - 1], sites, prime.(sites)) => 1)
    return nothing
end

function removeedges!(tensors::Vector{ITensor}, sites)
    tensors[1] *= onehot(Float64,
                         uniqueind(tensors[1], tensors[2], sites, prime.(sites)) => 1)
    tensors[end] *= onehot(Float64,
                           uniqueind(tensors[end], tensors[end - 1], sites, prime.(sites)) => 1)
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
    siteind(M, sp + 1) == tsite2 || error("Found wrong site")

    ctensor = _combinesiteinds(M[sp], M[sp + 1], tsite1, tsite2, csite)
    return MPS([M[1:(sp - 1)]..., ctensor, M[(sp + 2):end]...])
end

function combinesiteinds(M::MPS, csites; targetsites::Vector=siteinds(M))
    !hasedge(M) || error("MPS must not have edges")
    length(targetsites) == 2 * length(csites) || error("Length mismatch")
    for n in eachindex(csites)
        M = _combinesiteinds(M, targetsites[2 * n - 1], targetsites[2 * n], csites[n])
    end
    return M
end

_mklinks(dims) = [Index(dims[l], "Link,l=$l") for l in eachindex(dims)]

hasedge(M::MPS) = (length(inds(M[1])) == 3)

function _linkinds(M::MPS, sites::Vector{T}) where {T}
    N = length(M)
    if hasedge(M)
        links = T[]
        push!(links, uniqueind(M[1], M[2], sites))
        for n in 1:(N - 1)
            push!(links, commonind(M[n], M[n + 1]))
        end
        push!(links, uniqueind(M[end], M[end - 1], sites))
        return links
    else
        return linkinds(M)
    end
end

"""
Decompose a tensor into a set of indices by QR
"""
function split_tensor(tensor::ITensor, inds_list::Vector{Vector{Index{T}}}) where {T}
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
            replaceind!(M[n], links_old[n - 1], links_new[n - 1])
        end
    end
end

"""
To digits
"""
function tobin!(x::Int, xbin::Vector{Int})
    nbit = length(xbin)
    mask = 1 << (nbit - 1)
    for i in 1:nbit
        xbin[i] = (mask & x) >> (nbit - i)
        mask = mask >> 1
    end
end

function tobin(x::Int, R::Int)
    bin = zeros(Int, R)
    tobin!(x, bin)
    return bin
end

# Get bit at pos (>=0). pos=0 is the least significant digit.
_getbit(i, pos) = ((i & (1 << pos)) >> pos)

isascendingorder(x) = issorted(x; lt=isless)
isdecendingorder(x) = issorted(x; lt=Base.isgreater)

function kronecker_deltas(sitesin; sitesout=prime.(noprime.(sitesin)))
    N = length(sitesout)
    links = [Index(1, "Link,l=$l") for l in 0:N]
    M = MPO([delta(links[n], links[n + 1], sitesout[n], sitesin[n]) for n in 1:N])
    M[1] *= onehot(links[1] => 1)
    M[end] *= onehot(links[end] => 1)
    return M
end

"""
Match MPS/MPO to the given site indices

MPS:
The resultant MPS do not depends on the missing site indices.

MPO:
For missing site indices, identity operators are inserted.
"""
function matchsiteinds(M::Union{MPS,MPO}, sites)
    sites = noprime.(sites)
    positions = Int[findfirst(sites, s) for s in siteinds(M)]
    if length(M) > 1 && issorted(positions; lt=Base.isgreater)
        return matchsiteinds(MPO([M[n] for n in reverse(1:length(M))]), sites)
    end

    MSSTA.isascendingorder(positions) ||
        error("siteinds are not in ascending order!")

    M_edge = addedges(M)

    linkdims_org = dim.(linkinds(M_edge))
    linkdims_new = ones(Int, length(sites) + 1)
    for n in eachindex(M_edge)
        p = positions[n]
        linkdims_new[p] = linkdims_org[n]
        linkdims_new[p + 1] = linkdims_org[n + 1]
    end

    links = [Index(linkdims_new[l], "Link,l=$(l-1)") for l in eachindex(linkdims_new)]
    if typeof(M) == MPO
        tensors = [delta(links[n], links[n + 1]) * delta(sites[n], sites[n]')
                   for n in eachindex(sites)]
    elseif typeof(M) == MPS
        tensors = [delta(links[n], links[n + 1]) * ITensor(1, sites[n])
                   for n in eachindex(sites)]
    end

    links_old = linkinds(M_edge)
    for n in eachindex(M_edge)
        p = positions[n]
        tensor = copy(M_edge[n])
        replaceind!(tensor, links_old[n], links[p])
        replaceind!(tensor, links_old[n + 1], links[p + 1])
        if typeof(M) == MPO
            tensors[p] = permute(tensor, [links[p], links[p + 1], sites[p], sites[p]'])
        elseif typeof(M) == MPS
            tensors[p] = permute(tensor, [links[p], links[p + 1], sites[p]])
        end
    end

    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)

    return MPO(tensors)
end

#==
function _findtag(tag, sites::Vector{Index{T}}; only=true) where {T}
    idx = findall(hastags(tag), sites)
    if length(idx) == 0
        error("Not siteind with tag $tag found")
    elseif length(idx) > 1
        error("More than one siteind with tag $tag found")
    end
    return idx[1]
end
==#

"""
Find sites with the given tag

For tag = `x`, if `sites` contains an Index object with `x`, the function returns a vector containing only its positon.

If not, the function seach for all Index objects with tags `x=1`, `x=2`, ..., and return their positions.

If no Index object is found, an empty vector will be returned.
"""
function findallsites_by_tag(sites::Vector{Index{T}}; tag::String="x",
                             maxnsites::Int=1000)::Vector{Int} where {T}
    result = Int[]
    for n in 1:maxnsites
        tag_ = tag * "=$n"
        idx = findall(hastags(tag_), sites)
        if length(idx) == 0
            break
        elseif length(idx) > 1
            error("More siteinds with $(tag_) than one found")
        end
        push!(result, idx[1])
    end
    return result
end

function findallsiteinds_by_tag(sites; tag::String="x", maxnsites::Int=1000)
    positions = findallsites_by_tag(sites; tag=tag, maxnsites=maxnsites)
    return [sites[p] for p in positions]
end

asMPO(M::MPO) = M

function asMPO(tensors::Vector{ITensor})
    N = length(tensors)
    M = MPO(N)
    for n in 1:N
        M[n] = tensors[n]
    end
    return M
end

function asMPO(M::MPS)
    return asMPO(ITensors.data(M))
end

function asMPS(M::MPO)
    return MPS([t for t in M])
end

"""
Contract two adjacent tensors in MPO
"""
function combinesites(M::MPO, site1::Index, site2::Index)
    p1 = findsite(M, site1)
    p2 = findsite(M, site2)
    p1 === nothing && error("Not found $site1")
    p2 === nothing && error("Not found $site2")
    abs(p1 - p2) == 1 ||
        error("$site1 and $site2 are found at indices $p1 and $p2. They must be on two adjacent sites.")
    tensors = ITensors.data(M)
    idx = min(p1, p2)
    tensor = tensors[idx] * tensors[idx + 1]
    deleteat!(tensors, idx:(idx + 1))
    insert!(tensors, idx, tensor)
    return MPO(tensors)
end

function directprod(::Type{T}, sites, indices) where {T}
    length(sites) == length(indices) || error("Length mismatch between sites and indices")
    any(0 .== indices) && error("indices must be 1-based")
    R = length(sites)

    links = [Index(1, "Link,l=$l") for l in 0:R]
    tensors = ITensor[]
    for n in 1:R
        push!(tensors, onehot(links[n] => 1, links[n + 1] => 1, sites[n] => indices[n]))
    end
    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)
    return MPS(tensors)
end

function _find_target_sites(M::MPS; sitessrc=nothing, tag="")
    if tag == "" && sitessrc === nothing
        error("tag or sitesrc must be specified")
    elseif tag != "" && sitessrc !== nothing
        error("tag and sitesrc are exclusive")
    end

    # Set input site indices
    if tag != ""
        sites = siteinds(M)
        sitepos = findallsites_by_tag(sites; tag=tag)
        target_sites = [sites[p] for p in sitepos]
    elseif sitessrc !== nothing
        target_sites = sitessrc
        sitepos = Int[findsite(M, s) for s in sitessrc]
    end

    return sitepos, target_sites
end


function replace_siteinds_part!(M::MPS, sitesold, sitesnew)
    length(sitesold) == length(sitesnew) || error("Length mismatch between sitesold and sitesnew")

    for i in eachindex(sitesold)
        p = findsite(M, sitesold[i])
        if p === nothing
            error("Not found $(sitesold[i])")
        end
        replaceinds!(M[p], sitesold[i]=>sitesnew[i])
    end

    return nothing
end

"""
Connect two MPS's
ITensor objects are deepcopied.
"""
function _directprod(M1::MPS, Mx::MPS...)::MPS
    M2 = Mx[1]
    l = Index(1, "Link")
    tensors1 = [deepcopy(x) for x in M1]
    tensors2 = [deepcopy(x) for x in M2]
    tensors1[end] = ITensor(ITensors.data(last(tensors1)), [inds(last(tensors1))..., l])
    tensors2[1] = ITensor(ITensors.data(first(tensors2)), [l, inds(first(tensors2))...])

    M12 = MPS([tensors1..., tensors2...])
    if length(Mx) == 1
        return M12
    else
        return _directprod(M12, Mx[2:end]...)
    end
end
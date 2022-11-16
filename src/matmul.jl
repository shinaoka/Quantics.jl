
function matmul(a::MPS, b::MPS; kwargs...)
    N = length(a)
    mod(N, 2) == 0 || error("Length of a must be even")
    length(a) == length(b) || error("Length mismatch")
    halfN = N ÷ 2

    csites = [Index(4, "csite=$s") for s in 1:halfN]
    mpo_a = MSSTA.tompo_matmul(a, csites)

    b_ = MSSTA.combinesiteinds(b, csites)

    ab = apply(mpo_a, b_; kwargs...)
    res = MSSTA.splitsiteind(ab, siteinds(a))
    return res
end


function matmul_naive(a::MPS, b::MPS)
    N = length(a)
    mod(N, 2) == 0 || error("Length of a must be even")
    length(a) == length(b) || error("Length mismatch")
    halfN = N ÷ 2

    ab_tensors = ITensor[]
    sitesa = siteinds(a)
    sitesb = siteinds(b)
    linksa = MSSTA.links(a)
    linksb = MSSTA.links(b)
    newlinks = _mklinks([dim(linksa[2n]) * dim(linksb[2n]) for n in 1:halfN-1])
    newsites = [Index(4, "csite=$s") for s in 1:halfN]
    
    for n in 1:halfN
        a_ = a[2*n-1:2*n] # copy
        b_ = b[2*n-1:2*n]
        cind = Index(2, "Qubit")
        replaceind!(a_[2], sitesa[2*n], cind)
        replaceind!(b_[1], sitesb[2*n-1], cind)
        ab_ = a_[1] * ((a_[2] * b_[1]) * b_[2])
        if n == 1
           ab_ = permute(ab_, [sitesa[2n-1], sitesb[2n], linksa[2n], linksb[2n]])
           ab_ = ITensor(ITensors.data(ab_), [newsites[n], newlinks[n]])
        elseif n == halfN
           ab_ = permute(ab_, [linksa[2n-2], linksb[2n-2], sitesa[2n-1], sitesb[2n]])
           ab_ = ITensor(ITensors.data(ab_), [newlinks[n-1], newsites[n]])
        else
           ab_ = permute(ab_, [linksa[2n-2], linksb[2n-2], sitesa[2n-1], sitesb[2n], linksa[2n], linksb[2n]])
           ab_ = ITensor(ITensors.data(ab_), [newlinks[n-1], newsites[n], newlinks[n]])
        end
        push!(ab_tensors, ab_)
    end
    return splitsiteind(MPS(ab_tensors), siteinds(a))
end

function _tompo_matmul(t1::ITensor, t2::ITensor, sites, links, s)
    s1, s2 = sites
    l0, l1, l2 = links

    s1_new = Index(2, "site1")
    s2_new = Index(2, "site2")

    t1_ = copy(t1)
    replaceind!(t1_, s1, s1_new')

    t2_ = copy(t2)
    replaceind!(t2_, s2, s1_new)

    res = (t1_ * t2_) * delta(s2_new', s2_new)

    res = permute(res, [l0, l2, s1_new', s2_new', s1_new, s2_new])
    res = ITensor(ITensors.data(res), [l0, l2, s', s])
    return res
end


"""
Create tensors for matmul
"""
function tensors_matmul!(tensors::Vector{ITensor}, a::MPS, csites; targetsites=siteinds(a))
    sites = siteinds(a)
    length(targetsites) == 2*length(csites) || error("Length mismatch")

    #N = length(a)
    #halfN = N ÷ 2

    addedges!(a)
    linksa = _linkinds(a, sites)
    for n in eachindex(csites)
        startpos = findsite(a, targetsites[2*n-1])
        targetsites[2*n] == sites[startpos+1] || error("Not found")
        push!(tensors,
            _tompo_matmul(a[startpos], a[startpos+1], sites[startpos:startpos+1], linksa[startpos:startpos+2], csites[n])
        )
    end
    # Hack
    for (n, t) in enumerate(tensors)
        inds_ = inds(t)
        tensors[n] = permute(t, [inds_[1], inds_[3], inds_[4], inds_[2]])
    end
    removeedges!(a, sites)
    return nothing
end


function tompo_matmul(a::MPS, csites; targetsites=siteinds(a))
    tensors = ITensor[]
    tensors_matmul!(tensors, a, csites; targetsites=targetsites)
    M = MPO(tensors)
    removeedges!(M, csites)
    return M
end


"""
Create tensors for elementwise product
"""
function tensors_elementwiseprod!(tensors::Vector{ITensor}, a::MPS; targetsites=siteinds(a))
    Ntarget = length(targetsites)
    sites = siteinds(a)
    addedges!(a)

    for n in 1:Ntarget
        pos = findsite(a, targetsites[n])
        t = copy(a[pos])
        replaceind!(t, sites[pos], sites[pos]'')
        push!(tensors, t * delta(sites[pos], sites[pos]', sites[pos]''))
    end

    # Hack
    for (n, t) in enumerate(tensors)
        inds_ = inds(t)
        tensors[n] = permute(t, [inds_[1], inds_[3], inds_[4], inds_[2]])
    end

    removeedges!(a, sites)
    return nothing
end


function tompo_elementwiseprod(a::MPS; targetsites=siteinds(a))
    tensors = ITensor[]
    tensors_elementwiseprod!(tensors, a; targetsites=targetsites)
    M = MPO(tensors)
    removeedges!(M, siteinds(a))
    return M
end
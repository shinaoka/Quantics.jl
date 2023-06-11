function _single_tensor_flip()
    cval = [-1, 0]
    # (cin, cout, s', s)
    single_tensor = zeros(Float64, 2, 2, 2, 2)
    for icin in 1:2
        for a in 1:2
            out = -(a - 1) + cval[icin]
            icout = out < 0 ? 1 : 2
            b = mod(out, 2) + 1
            single_tensor[icin, icout, a, b] = 1
        end
    end
    return single_tensor
end

"""
f(x) = g(1 - x) = M * g(x)
"""
function flipop(sites::Vector{Index{T}}; rev_carrydirec=false, bc::Int=1)::MPO where {T}
    N = length(sites)
    abs(bc) == 1 || error("bc must be either 1, -1")
    N > 1 || error("MPO with one tensor is not supported")

    t = _single_tensor_flip()
    M = MPO(N)
    links = [Index(2, "Link,l=$l") for l in 1:(N + 1)]
    for n in 1:N
        M[n] = ITensor(t, (links[n], links[n + 1], sites[n]', sites[n]))
    end

    M[1] *= onehot(links[1] => 2)

    bc_tensor = ITensor([1.0, bc], links[end])
    M[N] = M[N] * bc_tensor

    if rev_carrydirec
        M = _reverse(M)
    end

    cleanup_linkinds!(M)

    return M
end

"""
f(x) = g(1 - x) = M * g(x) for x = 0, 1, ..., 2^R-1.

Note that x = 0, 1, 2, ..., 2^R-1 is mapped to x = 0, 2^R-1, 2^R-2, ..., 1.
"""
function reverseaxis(M::MPS; tag="", bc::Int=1, kwargs...)
    sites = siteinds(M)
    targetsites = sites
    if tag != ""
        targetsites = findallsiteinds_by_tag(sites; tag=tag)
    end

    transformer = matchsiteinds(flipop(targetsites; rev_carrydirec=true, bc=bc), sites)

    return apply(transformer, M; kwargs...)
end

"""
f(x) = g(x + shift) for x = 0, 1, ..., 2^R-1 and 0 <= shift < 2^R.
"""
function shiftaxis(M::MPS, shift::Int; tag="", bc::Int=1, kwargs...)
    sites = siteinds(M)
    targetsites = sites
    if tag != ""
        targetsites = findallsiteinds_by_tag(sites; tag=tag)
    end

    R = length(targetsites)
    nbc, shift_mod = divrem(shift, 2^R, RoundDown)

    transformer = matchsiteinds(shift_mpo(targetsites, shift_mod, bc), sites)
    transformer *= bc ^ nbc

    return apply(transformer, M; kwargs...)
end

"""
Multiply by exp(i θ x), where x = (x_1, ..., x_R)_2.
"""
function phase_rotation(M::MPS, θ::Float64; targetsites=nothing, tag="")
    sitepos, target_sites = _find_target_sites(M; sitessrc=targetsites, tag=tag)
    res = copy(M)

    nqbit = length(sitepos)
    for n in 1:nqbit
        p = sitepos[n]
        res[p] *= op("Phase", siteind(res, p); ϕ=θ * 2^(nqbit - n))
    end

    return noprime(res)
end

function _upper_lower_triangle(upper_or_lower::Symbol)::Array{Float64,4}
    upper_or_lower ∈ [:upper, :lower] || error("Invalid upper_or_lower $(upper_or_lower)")
    T = Float64
    t = zeros(T, 2, 2, 2, 2) # left link, right link, site', site

    t[1, 1, 1, 1] = one(T)
    t[1, 1, 2, 2] = one(T)
    
    if upper_or_lower == :upper
        t[1, 2, 1, 2] = one(T)
        t[1, 2, 2, 1] = zero(T)
    else
        t[1, 2, 1, 2] = zero(T)
        t[1, 2, 2, 1] = one(T)
    end

    # If a comparison is made at a higher bit, we respect it.
    t[2, 2, :, :] .= one(T)

    return t
end

"""
Create QTT for a upper/lower triangle matrix filled with one except the diagonal line
"""
function upper_lower_triangle_matrix(sites::Vector{Index{T}}, value::S; upper_or_lower::Symbol=:upper)::MPO where {T,S}
    upper_or_lower ∈ [:upper, :lower] || error("Invalid upper_or_lower $(upper_or_lower)")
    N = length(sites)

    t = _upper_lower_triangle(upper_or_lower)

    M = MPO(N)
    links = [Index(2, "Link,l=$l") for l in 1:(N + 1)]
    for n in 1:N
        M[n] = ITensor(t, (links[n], links[n + 1], sites[n]', sites[n]))
    end

    M[1] *= onehot(links[1] => 1)
    M[N] *= ITensor(S[0, value], links[N+1])

    return M
end

"""
Create MPO for cumulative sum in QTT

includeown = False
    y_i = sum_{j=1}^{i-1} x_j
"""
function cumsum(sites::Vector{Index}; includeown::Bool=false)
    includeown == False || error("includeown = True has not been implmented yet")
    return upper_triangle_matrix(sites, 1.0)
end

"""
Add new site indices to an MPS
"""
function asdiagonal(M::MPS, newsites; which_new="right", targetsites=nothing, tag="")
    which_new ∈ ["left", "right"] || error("Invalid which_new: left or right")
    sitepos, target_sites = MSSTA._find_target_sites(M; sitessrc=targetsites, tag=tag)
    length(sitepos) == length(newsites) ||
        error("Length mismatch: $(newsites) vs $(target_sites)")
    M_ = MSSTA.addedges(M)
    links = linkinds(M_)

    tensors = ITensor[]
    for p in 1:length(M)
        if !(p ∈ sitepos)
            push!(tensors, copy(M_[p]))
            continue
        end
        i = findfirst(x -> x == p, sitepos)
        s = target_sites[i]
        s1 = sim(s)
        ll, lr = links[p], links[p + 1]
        t = replaceind(M_[p], s => s1)
        if which_new == "right"
            tl, tr = factorize(delta(s1, s, newsites[i]) * t, ll, s)
        else
            tl, tr = factorize(delta(s1, s, newsites[i]) * t, ll, newsites[i])
        end
        push!(tensors, tl)
        push!(tensors, tr)
    end

    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)

    M_result = MPS(tensors)
    MSSTA.cleanup_linkinds!(M_result)
    return M_result
end

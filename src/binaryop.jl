"""
a * x + b * y, where a = 0/+/-1 and b = 0/+1/-1.
"""
function _binaryop_tensor(a::Int, b::Int, site_x::Index{T}, site_y::Index{T},
                          site_out::Index{T},
                          cin_on::Bool, cout_on::Bool, bc::Int) where {T}
    abs(a) <= 1 || error("a must be either 0, 1, -1")
    abs(b) <= 1 || error("b must be either 0, 1, -1")
    abs(bc) == 1 || error("bc must be either 1, -1")
    a + b != -2 || error("a = -1 and b = -1 not supported")

    cins = cin_on ? [-1, 0, 1] : [0]
    cinsize = length(cins)
    coutsize = cout_on ? 3 : 1
    tensor = Tensor(Float64, (cinsize, coutsize, 2, 2, 2))
    for (idx_cin, cin) in enumerate(cins), y in 0:1, x in 0:1
        res = a * x + b * y + cin
        if res >= 0
            cout = _getbit(abs(res), 1)
        else
            cout = -1
        end
        if cout_on
            tensor[idx_cin, cout + 2, x + 1, y + 1, (abs(res) & 1) + 1] = 1
        else
            tensor[idx_cin, 1, x + 1, y + 1, (abs(res) & 1) + 1] = (cout == 0 ? 1 : bc)
        end
    end
    link_in = Index(cinsize, "link_in")
    link_out = Index(coutsize, "link_out")
    return ITensor(tensor, [link_in, link_out, site_x, site_y, site_out]), link_in, link_out
end

"""
Create a tensor acting on a vector of sites.
"""
function binaryop_tensor_multisite(sites::Vector{Index{T}},
                                   coeffs::Vector{Tuple{Int,Int}},
                                   pos_sites_in::Vector{Tuple{Int,Int}},
                                   cin_on::Bool,
                                   cout_on::Bool,
                                   bc::Vector{Int}) where {T<:Number}

    # Check
    sites = noprime.(sites)
    nsites = length(sites)
    length(coeffs) == nsites || error("Length of coeffs does not match that of coeffs")
    length(pos_sites_in) == nsites ||
        error("Length of pos_sites_in does not match that of coeffs")

    sites_in = [Index(2, "site_dummy,n=$n") for n in eachindex(sites)]
    links_in = Index{T}[]
    links_out = Index{T}[]

    # First, we need to know the number of dummny indices for each site.
    ndumnyinds = zeros(Int, nsites)
    for n in 1:nsites
        for s in pos_sites_in[n]
            ndumnyinds[s] += 1
        end
    end

    res = ITensor(1)
    for n in 1:nsites
        res *= delta(sites[n], [setprime(sites_in[n], plev) for plev in 1:ndumnyinds[n]])
    end

    currentdummyinds = ones(Int, nsites)
    for n in 1:nsites
        sites_ab =
            setprime(sites_in[pos_sites_in[n][1]], currentdummyinds[pos_sites_in[n][1]]),
            setprime(sites_in[pos_sites_in[n][2]], currentdummyinds[pos_sites_in[n][2]])
        for i in 1:2
            currentdummyinds[pos_sites_in[n][i]] += 1
        end
        #sites_ab = setprime(sites_ab, n)
        t, lin, lout = _binaryop_tensor(
            coeffs[n]..., sites_ab..., sites[n]',
            cin_on, cout_on, bc[n])
        push!(links_in, lin)
        push!(links_out, lout)
        res *= t
    end
    linkin = Index(prod(dim.(links_in)), "linkin")
    linkout = Index(prod(dim.(links_out)), "linkout")
    res = permute(res, [links_in..., links_out..., prime.(sites)..., sites...])
    # Here, we swap sites and prime(sites)!
    res = ITensor(ITensors.data(res), [linkin, linkout, sites..., prime.(sites)...])
    return res
end

"""
Compute boundary conditions for `f` from those for `g`.
"""
#==
function _compute_bc_for_f(
    coeffs::Vector{Tuple{Int,Int}},
    pos_sites_in::Vector{Tuple{Int,Int}},
    bc_g::Vector{Int})::Vector{Int}

    N = length(coeffs)
    bc_f = ones(Int, N)
    for idx_f in 1:N
        for idx_g in 1:N
            for p in 1:2
                if idx_f == pos_sites_in[idx_g][p] && coeffs[idx_g][p] != 0
                    bc_f[idx_f] *= bc_g[idx_g]
                end
            end
        end
    end

    return bc_f
end
==#

"""
Construct an MPO representing a selector associated with binary operations.

We describe the functionality for length(coeffs) = 2 (nsites_bop).
In this case, site indices are split into a list of chuncks of nsites_bop sites.

Binary operations are applied to each chunck and the direction of carry is forward (rev_carrydirec=true)
or backward (rev_carrydirec=false).

Assumed rev_carrydirec = true, we consider a two-variable g(x, y), which is quantized as
x = (x_1 ... x_R)_2, y = (y_1 ... y_R)_2.

We now define a new function by binary operations as
f(x, y) = g(a * x + b * y + s1, c * x + d * y + s2),
where a, b, c, d = +/- 1, 0, and s1, s1 are arbitrary integers.

Let us first explain the case of s1, s2 = 0 (no shift).

The transform from `g` to `f` can be represented as an MPO:
f(x_1, y_1, ..., x_R, y_R) = M(x_1, y_1, ...; x'_1, y'_1, ...) f(x'_1, y'_1, ..., x'_R, y'_R).

The MPO `M` acts a selector: The MPO selects values from `f` to form `g`.

For rev_carrydirec = false, the returned MPO represents
f(x_R, y_R, ..., x_1, y_1) = M(x_R, y_R, ...; x'_R, y'_R, ...) f(x'_R, y'_R, ..., x'_1, y'_1).

`bc` is a vector of boundary conditions for each arguments of `g` (not of `f`).
"""
function binaryop_mpo_shift(sites::Vector{Index{T}},
    coeffs::Vector{Tuple{Int,Int}},
    pos_sites_in::Vector{Tuple{Int,Int}},
    shift::Vector{Tuple{Int,Int}};
    rev_carrydirec=false,
    bc::Union{Nothing,Vector{Int}}=nothing) where {T<:Number}

    # Number of variables involved in transformation
    nsites_bop = length(coeffs)

    length(pos_sites_in) == nsites_bop || error("Length of pos_sites_in does not match that of coeffs")
    length(shift) == nsites_bop || error("Length of shift does not match that of coeffs")

    # f(x, y) = g(a * x + b * y + s1, c * x + d * y + s2)
    #         = h(a * x + b * y,      c * x + d * y),
    # where h(x, y) = g(x + s1, y + s2).
    # The transformation is taken place in this order: g -> h -> f.

    # g -> h
    #M_g_to_h = shift_mpo(sites, shift, rev_carrydirec=rev_carrydirec, bc=bc[i])

    # h -> f
    M_h_to_f = binaryop_mpo(sites, coeffs, pos_sites_in, rev_carrydirec=rev_carrydirec, bc=bc)
end

function binaryop_mpo(sites::Vector{Index{T}},
                      coeffs::Vector{Tuple{Int,Int}},
                      pos_sites_in::Vector{Tuple{Int,Int}};
                      rev_carrydirec=false,
                      bc::Union{Nothing,Vector{Int}}=nothing) where {T<:Number}
    # Number of variables involved in transformation
    nsites_bop = length(coeffs)

    if bc === nothing
        bc = ones(Int64, nsites_bop) # Default: periodic boundary condition
    end

    # First check transformations with -1 and -1; e.g., (a, b) = (-1, -1)
    # These transformations are not supported in the backend.
    # To support them, we need to flip the sign of coeffs as follows:
    #  f(x, y) = h(x + y, c * x + d * y) = g(-x-y, c * x + d * y),
    # where h(x, y) = g(-x, y).
    # The transformation is taken place in this order: g -> h -> f.
    sign_flips = [coeffs[n][1] == -1 && coeffs[n][2] == -1 for n in 1:length(coeffs)]
    coeffs_ = [(sign_flips[i] ? abs.(coeffs[i]) : coeffs[i]) for i in eachindex(coeffs)]

    # For g->h
    M = _binaryop_mpo(sites, coeffs_, pos_sites_in; rev_carrydirec=rev_carrydirec, bc=bc)

    # For h->f
    for i in 1:nsites_bop
        if !sign_flips[i]
            continue
        end
        M_ = bc[i] * flipop(sites[i:nsites_bop:end], rev_carrydirec=rev_carrydirec, bc=bc[i])
        M = apply(M, matchsiteinds(M_, sites), alg="naive", cutoff=1e-25)
    end

    return M
end


# Limitation: a = -1 and b = -1 not supported. The same applies to (c, d).
function _binaryop_mpo(sites::Vector{Index{T}},
                      coeffs::Vector{Tuple{Int,Int}},
                      pos_sites_in::Vector{Tuple{Int,Int}};
                      rev_carrydirec=false,
                      bc::Union{Nothing,Vector{Int}}=nothing) where {T<:Number}
    nsites = length(sites)
    nsites_bop = length(coeffs)
    ncsites = nsites ÷ nsites_bop
    length(pos_sites_in) == nsites_bop ||
        error("Length mismatch between coeffs and pos_sites_in")
    if bc === nothing
        bc = ones(Int64, nsites_bop) # Default: periodic boundary condition
    end

    links = [Index(3^nsites_bop, "link=$n") for n in 0:ncsites]
    links[1] = Index(1, "link=0")
    links[end] = Index(1, "link=$ncsites")

    tensors = ITensor[]
    sites2d = reshape(sites, nsites_bop, ncsites)
    for n in 1:ncsites
        sites_ = sites2d[:, n]
        cin_on = rev_carrydirec ? (n != ncsites) : (n != 1)
        cout_on = rev_carrydirec ? (n != 1) : (n != ncsites)
        tensor = binaryop_tensor_multisite(sites_,
                                           coeffs,
                                           pos_sites_in,
                                           cin_on,
                                           cout_on,
                                           bc)
        lleft, lright = links[n], links[n + 1]
        if rev_carrydirec
            replaceind!(tensor, firstind(tensor, "linkout") => lleft)
            replaceind!(tensor, firstind(tensor, "linkin") => lright)
        else
            replaceind!(tensor, firstind(tensor, "linkin") => lleft)
            replaceind!(tensor, firstind(tensor, "linkout") => lright)
        end

        inds_list = [[lleft, sites_[1]', sites_[1]]]
        for m in 2:(nsites_bop - 1)
            push!(inds_list, [sites_[m]', sites_[m]])
        end
        push!(inds_list, [lright, sites_[nsites_bop]', sites_[nsites_bop]])

        #@show inds(tensor)
        #@show inds_list
        #for t in split_tensor(tensor, inds_list)
            #@show inds(t)
        #end
        tensors = vcat(tensors, split_tensor(tensor, inds_list))
    end

    removeedges!(tensors, sites)

    M = truncate(MPO(tensors); cutoff=1e-25)
    cleanup_linkinds!(M)

    return M
end


"""
For given function `g(x)` and shift `s`, construct an MPO representing `f(x) = g(x + s)`.
x: 0, ..., 2^R - 1
0 <= s <= 2^R - 1

We assume that left site indices correspond to significant digits
"""
function shift_mpo(sites::Vector{Index{T}}, shift::Int; rev_carrydirec=false, bc::Int=1) where {T<:Number}
    R = length(sites)
    0 <= shift <= 2^R-1 || error("Invalid shift")

    ys = MSSTA.tobin(shift, R)

    links = Index{T}[]
    tensors = ITensor[]

    for n in 1:R
        cin_on = n != R
        cout_on = n != 1
        sitey = Index(2, "Qubit, y")
        t, link_in, link_out = MSSTA._binaryop_tensor(1, 1, sites[n]', sitey, sites[n], cin_on, cout_on, bc)
        t *= onehot(sitey => ys[n]+1)
        if n < R
            push!(links, Index(dim(link_in), "link=$n"))
            replaceind!(t, link_in => links[end])
        end
        if n > 1
            replaceind!(t, link_out => links[n-1])
        end
        if n == 1
            t *= onehot(link_out => 1)
        elseif n == R
            t *= onehot(link_in => 1)
        end

        push!(tensors, t)
    end

    MPO(tensors)
end
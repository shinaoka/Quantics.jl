using .TensorCrossInterpolation
import .TensorCrossInterpolation: TensorCI, CachedFunction, TensorCI2
import .TensorCrossInterpolation as TCI

function TCItoMPS(tci::Union{TensorCI{T},TensorCI2{T}}, sites=nothing) where {T}
    tensors = TCI.tensortrain(tci)
    ranks = TCI.rank(tci)
    N = length(tensors)
    localdims = [size(t, 2) for t in tensors]

    if sites === nothing
        sites = [Index(localdims[n], "n=$n") for n in 1:N]
    else
        all(localdims .== dim.(sites)) ||
            error("ranks are not consistent with dimension of sites")
    end

    linkdims = [[size(t, 1) for t in tensors]..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N]

    tensors_ = [ITensor(deepcopy(tensors[n]), links[n], sites[n], links[n + 1])
                for n in 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return MPS(tensors_)
end

abstract type AbstractAdaptiveQTTNode end

struct AdaptiveQTT{T<:Number,DIM} <: AbstractAdaptiveQTTNode
    f::Function
    tci::TensorCI{T}
    ranks::Vector{Int}
    errors::Vector{Float64}
    prefix::Vector{QuanticsInd}
end

function Base.show(io::IO, obj::AdaptiveQTT{T,DIM}) where {T,DIM}
    prefix = convert.(Int, obj.prefix)
    println(io,
            " "^length(obj.prefix) *
            "AdaptiveQTT: prefix=$(prefix), rank=$(maximum(obj.ranks))")
end

struct AdaptiveQTTInternalNode{T<:Number,DIM} <: AbstractAdaptiveQTTNode
    children::Vector{AbstractAdaptiveQTTNode}
    prefix::Vector{QuanticsInd}
end

function Base.show(io::IO, obj::AdaptiveQTTInternalNode{T,DIM}) where {T,DIM}
    #prefix = convert.(Int, obj.prefix)
    #println(io, " "^length(obj.prefix) * "AdaptiveQTTInternalNode: prefix=$(prefix)")
    for c in obj.children
        Base.show(io, c)
    end
end

function construct_adaptiveqtt(::Type{T}, ::Val{DIM}, f::Function, R::Int; maxiter=100,
                               prefix=QuanticsInd[], kwargs...) where {T,DIM}
    localdim = 2^DIM
    lenprefix = length(prefix)

    function _arg_conv(qs::Vector{Int})
        qs_ = [vcat(prefix, QuanticsInd{DIM}.(qs))...]
        idx = quantics_to_index(qs_)
        return (idx .- 1) .* (0.5^R)
    end

    localdims = fill(localdim, R - lenprefix)
    fc = TCI.CachedFunction{T}(x -> f(_arg_conv(x)), localdims)
    tci, ranks, errors = crossinterpolate(T, fc, localdims,
                                           ones(Int, R - lenprefix);
                                           maxiter=maxiter,
                                           kwargs...)

    if maximum(ranks) < maxiter ÷ 2
        return AdaptiveQTT{T,DIM}(f, tci, Vector{Int}(ranks), Vector{Float64}(errors),
                                  prefix)
    end

    # If the rank is too large
    children = AbstractAdaptiveQTTNode[]
    for ic in 1:localdim
        c = construct_adaptiveqtt(T, Val(DIM), f, R; maxiter=maxiter,
                                  prefix=vcat(prefix, QuanticsInd{DIM}(ic)), kwargs...)
        push!(children, c)
    end
    return AdaptiveQTTInternalNode{T,DIM}(children, prefix)
end

function asmps(qatt::AdaptiveQTT{T,DIM}, sites; kwargs...)::MPS where {T,DIM}
    lenprefix = length(qatt.prefix)
    M = TCItoMPS(qatt.tci)
    replace_siteinds!(M, sites[(lenprefix + 1):end])

    if lenprefix == 0
        return M
    end

    M_prefix = directprod(T, sites[1:lenprefix], convert.(Int, qatt.prefix))
    res = _directprod(M_prefix, M)
    return res
end

function asmps(qatt::AdaptiveQTTInternalNode{T,DIM}, sites; kwargs...)::MPS where {T,DIM}
    children = MPS[asmps(c, sites) for c in qatt.children]
    M = children[1]
    for c in 2:length(children)
        M += children[c]
        truncate!(M; kwargs...)
    end

    return M
end

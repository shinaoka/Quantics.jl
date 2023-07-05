
# A valid tag should not contain "=".
_valid_tag(tag::String)::Bool = !occursin("=", tag)


"""
Find sites with the given tag

For tag = `x`, if `sites` contains an Index object with `x`, the function returns a vector containing only its positon.

If not, the function seach for all Index objects with tags `x=1`, `x=2`, ..., and return their positions.

If no Index object is found, an empty vector will be returned.
"""
function findallsites_by_tag(sites::Vector{Index{T}}; tag::String="x",
                             maxnsites::Int=1000)::Vector{Int} where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")
    result = Int[]
    for n in 1:maxnsites
        tag_ = tag * "=$n"
        idx = findall(hastags(tag_), sites)
        if length(idx) == 0
            break
        elseif length(idx) > 1
            error("Found more than one siteind with $(tag_)!")
        end
        push!(result, idx[1])
    end
    return result
end

function findallsiteinds_by_tag(sites; tag::String="x", maxnsites::Int=1000)
    _valid_tag(tag) || error("Invalid tag: $tag")
    positions = findallsites_by_tag(sites; tag=tag, maxnsites=maxnsites)
    return [sites[p] for p in positions]
end
@doc raw"""
Discrete mesh for a ``N`` dimensional coordinate system.

The origin of the mesh can be arbitrary, defaulting to 0.
The linear size of the mesh is ``2^R``.
"""
struct DiscreteMesh{N}
    R::Int
    origin::NTuple{N,Int}
end

Base.iterate(s::DiscreteMesh) = (s, nothing)

# 'continue' iteration
Base.iterate(s::DiscreteMesh, state) = nothing  # No more values to iterate

# Create a discrete mesh
function DiscreteMesh{N}(R::Int) where {N}
    return DiscreteMesh{N}(R, Tuple(zeros(Int, N)))
end

# Convert an index in the mesh to the coordinate in the original coordinate system
function originalcoordinate(mesh::DiscreteMesh, index::NTuple{N,Int}) where {N}
    all(1 .<= index .<= 2^mesh.R) || throw(BoundsError(index, 1, 2^mesh.R))
    return (index .- 1) .+ mesh.origin
end

# Convert a coordinate in the original coordinate system to the index in the mesh
function meshindex(mesh::DiscreteMesh, coordinate::NTuple{N,Int}) where {N}
    return (coordinate .- mesh.origin) .+ 1
end


# Convert an index in the mesh to qubit indices
#function qubitindex(mesh::DiscreteMesh, index::NTuple{N,Int}) where {N}
    #return
#end
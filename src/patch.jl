"""
Contract M1 and M2, and return the result as an MPO.
"""
function ITensors.contract(
    ::ITensors.Algorithm"fit", M1::MPO, M2::MPO; init=M2, kwargs...
  )::MPO
    t1 = TreeTensorNetwork([M1[v] for v in eachindex(M1)])
    t2 = TreeTensorNetwork([M2[v] for v in eachindex(M2)])
    t0 = TreeTensorNetwork([init[v] for v in eachindex(M2)])
    t12 = contract(t1, t2; alg="fit", init=t0, kwargs...)
    return MPO([t12[v] for v in eachindex(M1)])
end
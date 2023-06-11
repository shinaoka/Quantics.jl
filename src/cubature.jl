"""
Adaptive cubature of a function f on [0, 1]^D using QTT

T: type of the result
D: dimension of the domain
f: function to integrate, which takes a vector of length D as input
R: Number of quantics indices in each dimension
"""
function adaptive_cubature(::Type{T}, D::Int, f, R::Int;
    tolerance::Float64=1e-12, maxbonddim::Int=100, verbosity::Int=0, loginterval::Int=10, ncheckhistory=1) where {T}
    tol_rat = 0.1

    if D == 1
        @memoize function f__(x)
            return f(x[1])
        end
        return adaptive_quadrature(T, f__, R;
            tolerance=tol_rat * tolerance, maxbonddim=maxbonddim, verbosity=verbosity, loginterval=10, ncheckhistory=1)
    end

    # 1D function after the rest of the dimensions are integrated out
    @memoize function f_(x::Float64)::T
        res = adaptive_cubature(
                T, D-1,
                xrest::Float64->f([x, xrest...]),
                R;
                tolerance=tol_rat * tolerance,
                maxbonddim=maxbonddim,
                verbosity=verbosity,
                loginterval=10,
                ncheckhistory=1
            )
        #println("$x $res")
        return res
    end
    return adaptive_quadrature(T, f_, R; tolerance=tolerance, maxbonddim=maxbonddim, verbosity=verbosity, loginterval=loginterval, ncheckhistory=3)
end


"""
Adaptive integration of a function f on [0, 1] using QTT
"""
function adaptive_quadrature(::Type{T}, f, R::Int; tolerance::Float64=1e-12, maxbonddim::Int=100, verbosity::Int=0, loginterval::Int=10, ncheckhistory=3)::T where{T}
    g = DiscretizedGrid{1}(R) # Grid on [0, 1]

    function f_(x::Vector{Int})::T 
        x_ = originalcoordinate(g, QubitInd.(x))
        return f(x_[1])
    end

    localdims = fill(2, R)
    tree = adaptivetci(T, f_, localdims; maxbonddim=maxbonddim, tolerance=tolerance, verbosity=verbosity, maxnleaves=100000, loginterval=loginterval, ncheckhistory)

    return sum(tree)/2.0^R
end
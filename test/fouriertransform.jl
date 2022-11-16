using Test
using MultiScaleSpaceTimes
using ITensors

@testset "fouriertransform.jl" begin
    @testset "_qft" for targetfunc in [MultiScaleSpaceTimes._qft_ref, MultiScaleSpaceTimes._qft], sign in [1, -1], nbit in [1, 2, 3]
    #@testset "_qft" for targetfunc in [MultiScaleSpaceTimes._qft_ref], sign in [1], nbit in [1]
        N = 2^nbit
        
        sites = siteinds("Qubit", nbit)
        M = targetfunc(sites; sign=sign)

        # Return the bit of an integer `i` at the position `pos` (`pos=1` is the least significant digit).
        bitat(i, pos) = ((i & 1<<(pos-1))>>(pos-1))

        # Input function `f(x)` is 1 only at xin otherwise 0.
        for xin in [0, 1, N-1]
            # From left to right (x_1, x_2, ...., x_Q)
            mpsf = MPS(sites, collect(string(bitat(xin, pos)) for pos in nbit:-1:1))
            mpsg = reduce(*, noprime(contract(M, mpsf)))
    
            # Values of output function
            # (k_Q, ...., k_1)
            outfunc = vec(Array(mpsg, sites))
    
            @test outfunc ≈ [exp(sign * im * 2π * y * xin/N)/sqrt(N) for y in 0:(N-1)]
        end
    end
end

using Test
using MSSTA
using ITensors

@testset "fouriertransform.jl" begin
    @testset "_qft" for targetfunc in [MSSTA._qft_ref, MSSTA._qft], sign in [1, -1], nbit in [1, 2, 3]
        N = 2^nbit
        
        sites = siteinds("Qubit", nbit)
        M = targetfunc(sites; sign=sign)

        # Return the bit of an integer `i` at the position `pos` (`pos=1` is the least significant digit).
        bitat(i, pos) = ((i & 1<<(pos-1))>>(pos-1))

        # Input function `f(x)` is 1 only at xin otherwise 0.
        for xin in [0, 1, N-1]
            # From left to right (x_1, x_2, ...., x_Q)
            
            mpsf = MPS(sites, collect(string(bitat(xin, pos)) for pos in nbit:-1:1))
            mpsg = reduce(*, apply(M, mpsf))
    
            # Values of output function
            # (k_Q, ...., k_1)
            outfunc = vec(Array(mpsg, sites))
    
            @test outfunc ≈ [exp(sign * im * 2π * y * xin/N)/sqrt(N) for y in 0:(N-1)]
        end
    end

    function _ft_1d_ref(X, sign)
        N = length(X)
        Y = zeros(ComplexF64, N)
        for k in 1:N
            for x in 1:N
                Y[k] += exp(sign * im * 2π * (k-1) * (x-1)/N) * X[x]
            end
        end
        Y ./= sqrt(N)
        return Y
    end

    @testset "fouriertransform_1d" for sign in [1, -1], nbit in [2, 3, 4]
        N = 2^nbit
        
        sitesx = [Index(2, "Qubit,x=$x") for x in 1:nbit]
        sitesk = [Index(2, "Qubit,k=$k") for k in 1:nbit]

        # X(x)
        X = randomMPS(sitesx)
        X_vec = Array(reduce(*, X), reverse(sitesx))
    
        # Y(k)
        Y = MSSTA.fouriertransform(X; sign=sign, tag="x", sitesdst=sitesk)

        Y_vec_ref = _ft_1d_ref(X_vec, sign)
        Y_vec = vec(Array(reduce(*, Y), reverse(sitesk)))
    
        @test Y_vec ≈ Y_vec_ref
    end

    function _ft_2d_ref(F::Matrix, sign)
        N = size(F, 1)
        G = zeros(ComplexF64, N, N)
        for ky in 1:N, kx in 1:N
            for y in 1:N, x in 1:N
                G[kx, ky] +=
                    exp(sign * im * 2π * (kx-1) * (x-1)/N) *
                    exp(sign * im * 2π * (ky-1) * (y-1)/N) * F[x, y]
            end
        end
        G ./= N
        return G
    end

    @testset "fouriertransform_2d" for sign in [1, -1], nbit in [2, 3]
        N = 2^nbit
        
        sitesx = [Index(2, "Qubit,x=$x") for x in 1:nbit]
        sitesy = [Index(2, "Qubit,y=$y") for y in 1:nbit]
        siteskx = [Index(2, "Qubit,kx=$kx") for kx in 1:nbit]
        sitesky = [Index(2, "Qubit,ky=$ky") for ky in 1:nbit]

        sitesin  = collect(Iterators.flatten(zip(sitesx, sitesy)))

        # F(x, y)
        # F(x_1, y_1, ..., x_R, y_R)
        F = randomMPS(sitesin)
        F_mat = reshape(Array(reduce(*, F), vcat(reverse(sitesx), reverse(sitesy))), N, N)
    
        # G(kx,ky)
        # G(kx_R, ky_R, ..., kx_1, ky_1)
        G_ = MSSTA.fouriertransform(F; sign=sign, tag="x", sitesdst=siteskx)
        G = MSSTA.fouriertransform(G_; sign=sign, tag="y", sitesdst=sitesky)

        G_mat_ref = _ft_2d_ref(F_mat, sign)
        G_mat = reshape(Array(reduce(*, G), vcat(reverse(siteskx), reverse(sitesky))), N, N)
    
        @test G_mat ≈ G_mat_ref
    end

end

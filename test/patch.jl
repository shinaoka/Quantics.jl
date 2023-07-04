using Test
import Quantics
using ITensors
using ITensorTDVP
using Random

@testset "patch.jl" begin
    @testset "Contract MPO-MPO" begin
        Random.seed!(1234)
        nbit = 5
        sites = siteinds("Qubit", nbit)
        M1 = randomMPO(sites) + randomMPO(sites)
        M2 = randomMPO(sites) + randomMPO(sites)
      
        # The function `apply` does not work correctly with the mapping-MPO-to-MPS trick.
        M1 = replaceprime(M1, 1=>2, 0=>1)
      
        M2_ = MPS(length(sites))
        for n in eachindex(sites)
          M2_[n] = M2[n]
        end
      
        M12_ref = contract(M1, M2; alg="naive")
        M12 = Quantics._contract_fit(M1, M2_)
        t12_ref = Array(reduce(*, M12_ref), sites, setprime(sites, 2))
        t12 = Array(reduce(*, M12), sites, setprime(sites, 2))
        @test maximum(abs, t12 .- t12_ref) < 1e-12 * maximum(abs, t12_ref) 

        M12_2 = Quantics._contract_fit(M1, M2)
        t12_2 = Array(reduce(*, M12_2), sites, setprime(sites, 2))
        @test maximum(abs, t12_2 .- t12_ref) < 1e-12 * maximum(abs, t12_ref) 
    end
end

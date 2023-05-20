
"""
Create a QTT representing exp(a*x) on [0, 1)

exp(-a*x) = prod_{n=1}^R exp(a * 2^(-n) * x_n)
"""
function expqtt(sites, a::Float64)
    R = length(sites)
    links = [Index(1, "Link,l=$l") for l in 0:R]
    tensors = ITensor[]
    for n in 1:R
        push!(tensors,
              ITensor([1.0, exp(a * (0.5^n))], links[n], links[n + 1], sites[n]))
    end
    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)
    return MPS(tensors)
end
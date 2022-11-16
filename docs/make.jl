using MSSTA
using Documenter

DocMeta.setdocmeta!(MSSTA, :DocTestSetup, :(using MSSTA); recursive=true)

makedocs(;
    modules=[MSSTA],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://github.com/shinaoka/MSSTA.jl/blob/{commit}{path}#{line}",
    sitename="MSSTA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

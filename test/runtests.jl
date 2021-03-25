cd(@__DIR__)

tests = readdir()
filter!(x -> x != "runtests.jl", tests)

res = map(tests) do t
    include(t)
end

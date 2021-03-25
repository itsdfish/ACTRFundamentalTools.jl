# set the working directory to the directory in which this file is contained
cd(@__DIR__)
println("Installing and compiling dependencies...")
println("This will take several minutes...")
# load the package manager
using Pkg
Pkg.activate("..")
# initialize project environment
Pkg.instantiate()
# precompile the packages
Pkg.precompile()
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
println("Installing and compiling dependencies...")
println("This will take several minutes...")
println("If prompted to install Anaconda, press Y to accept...")
# load the package manager
using Pkg
Pkg.add("IJulia")
Pkg.precompile()
# activate the project environment
Pkg.activate("")
# initialize project environment
Pkg.instantiate()
Pkg.update()
Pkg.precompile()
# Installation

## Julia

Download Julia 1.6 or higher from https://julialang.org/downloads/

## VSCode IDE

Optionally, download VSCode for editing and highlighting from https://code.visualstudio.com/download
and install the Julia plugin for VSCode per instructions outlined at https://www.julia-vscode.org/docs/dev/gettingstarted/#Installation-and-Configuration-1.

## Package Dependencies

Once Julia is intalled, open it and type into the command line:

```julia
include("my_path/ACTRFundamentalTools/install_packages.jl")
```
where `my_path` is the path to the code on your machine. Installation may take several minutes.

You might be propmpted to install Anaconda for the Jupyter notebooks. Confirm installation. 

You may optionally confirm proper installation by running unit tests for the code base with the following command:

```julia 
] test
```
# Running the code

A sub-folder for each model is located in the parent folder called models. The run file for each model 
begins with "Run". Run the file with `include("path_to_model_run_file/model_file.jl")` or opening and running in VSCode. 

# Running the notebook

Open Julia and enter the following command:

```julia
using IJulia; notebook(dir=pwd())
```

You pay replace `pwd()` with a specific directory if desired. Once Jupyter has launched in your browswer, use the file navigator to find `ACTRFundamentalTools/Table_of_Conents.ipynb` and click to start. A table of contents with hyperlinks for each model will open in your browswer. 



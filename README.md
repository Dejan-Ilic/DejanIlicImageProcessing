# DejanIlicImageProcessing

## Julia set up
- Install the latest version of Julia
- Open the julia REPL 
- To download this project, type `]`, followed by `dev https://github.com/Dejan-Ilic/DejanIlicImageProcessing`. This adds this project to `~/.julia/dev`.
- Press backspace to exit dev mode and return to the julia REPL
- Enter `using Pkg; Pkg.add("Pluto")`
- After Pluto's installation finishes, type `using Pluto; Pluto.run()`. This should open a browser tab. Leave it open.

## Usage
- Copy the `base_notebook.jl` file from `~/.julia/dev/DejanIlicImageProcessing/notebooks` to an easy to find location on your computer. Rename it if you want to.
- Open it with any text editor and change `push!(LOAD_PATH, *your dev folder location*)` (line 11 or so). Save. 
- Now return to the Pluto browser tab and open the copy of `base_notebook.jl` you have just made.

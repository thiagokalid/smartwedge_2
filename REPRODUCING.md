# Reproducing our results

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

```
git clone https://github.com/thiagokalid/acoustic-lens-pipe-inspection.git
```

or downloading a zip archive from the provided DOI in the `README.md`.

All source code used to generate the results and figures in the paper are in
the `scripts` folder. There you can find the `.py` scripts that perform the calculations and generate the
figures and results presented in the paper. A set of suplementary modules was developed under the `pipe_lens` folder. 

All the generated figures are stored in `figures` folder in `.pdf` format.

The data required to generate the results are available at the Zenodo [repository](https://doi.org/10.5281/zenodo.15115255).

## Setting up your environment

You'll need a working Python 3 environment with all packages described in `requirements.txt`.

Instead of manually installing them, they can all be automatically installed
using a virtual environments (venv):

1. Inside the cloned repository (or an unzipped version), create a new virtual by
   ```
   python3 -m venv C:\path\to\new\virtual\environment
   ```
   
1. Activate the new environment by running:
    ```
   source .venv_folder/bin/activate
   ```

1. Finally, install all packages from the requirements.txt:
   ```
   python3 -m pip install -r requirements.txt
   ```


## Generating the results from the paper

All results and their associated figures are created by the notebooks in the
`scripts` folder. Each script is named after the correspondent figure numbering presented in the
paper manuscript. By running the correspondent script, a new figure will be generated under the `figures`. Since all 
figures presented in the paper were previously generated, it might appear that there wasn't a new file. Some scripts
might take a while to run and a good amount of RAM, so be aware.

With the environment activated, run each script by:

```
python3 scripts/script_name.py
```
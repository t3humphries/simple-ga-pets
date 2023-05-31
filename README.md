# Source code for the paper: Differentially Private Simple Genetic Algorithms
## Hardware Requirements
To run all experiments in the paper, including grid searches, over a week is required on large machines with over 160 CPUs.
However, a single run of the SimpleGA takes less than a minute on average quad-core hardware. 
We thus give two interfaces to our code: 
1) Run DP Simple GA 
2) Reproduce Experiments. 

## Build Instructions
### Docker:
We will upload a docker container (by June 8th with permission from the chairs) with all python dependencies installed.
### Build From Source:
#### Dependencies:
- pandas==1.1.4
- scipy==1.5.4
- matplotlib==3.3.3
- numpy==1.19.4
- scikit_learn==0.24.1
- ray==2.1.0

Once the dependencies are installed, make sure to build the cython code using
<pre><code>
    cd pam_algo
    python3 setup.py install
    </code></pre>

## Running the Simple GA
We will provide a CLI (by June 8th with permission from the chairs) to run our GA with arguments such as the dataset, problem, and all hyperparameters.

## Reproducing Experiments
We provide a bash script for running the code run.sh.
The main.py file takes 3 args:

```--dataset_name``` which can take any value in [Adult, Credit, Mushrooms, Spam\]$$

```--problem_name``` which can be LR or KM_* where * can be any number of medians (less than public set size) we use k=16

```--experiment``` with args descibed below

'generations' - Run the simple GA over various generations and epsilons

'pg_generations' - Run PrivGene over various generations and epsilons

'grid' - Run a grid search over all params of the simple GA

'non-priv_grid' - Run a grid search over all params of the simple GA - Non-private

'grid_rerun' - Run 100 runs of the best grid parameters

'privGene' - Run the default version of PrivGene

'jones' - Run Jones et al.'s $k$ median approach

'selection' - Test various selection and composition techniques on the simple GA

'utility' - Test the effect of distance matrix clipping

'privLocal' - Run the default version of PrivLocal

'privLocal_mod' - Run the modified version of PrivLocal

'privEEM' - Run the default version of PrivEEM

'privEEM_mod' - Run the modified version of PrivEEM

'gupta' - Run Gupta et al's Local Search algorithm

'gupta_mod' - Run our modified version of Gupta et al's Local Search algorithm

### Plotting
We included the jupiter notebook used to create the plots for our paper. We note that the experiments will need to be re-run for this script to work.

## Datasets
Datasets can be found here:
- Adult:  https://archive.ics.uci.edu/ml/datasets/adult
- Credit: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Spam: https://archive.ics.uci.edu/ml/datasets/spambase
- Mushrooms: https://archive.ics.uci.edu/ml/datasets/mushroom

Pre-processing is done by running dataset_maker.py
Pickle files of the processed datasets will be included in the docker.

## PrivGene Notes
Priv Gene Code from https://sourceforge.net/projects/privgene/

Last Update: 2014-07-09

Initial Converstion to python by Lee and Kifer in: https://arxiv.org/pdf/1808.09501.pdf

Code: https://github.com/ppmlguy/DP-AGD

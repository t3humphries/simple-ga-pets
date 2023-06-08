# Source code for the paper: Differentially Private Simple Genetic Algorithms
## Hardware Requirements
To run all experiments in the paper, including grid searches, over a week is required on large machines with over 160 CPUs.
However, a single run of the SimpleGA takes less than a minute on average quad-core hardware. 
We thus give two interfaces to our code: 
1) Run DP Simple GA (our_dpga.py)
2) Reproduce Experiments (main.py). 

## Build Instructions
### Docker:
You can find a docker image (with all the necessary dependencies installed) here:
https://hub.docker.com/r/t3humphr/dp_simple_ga

You can pull directly from dockerhub with:
<pre><code>docker pull t3humphr/dp_simple_ga
</code></pre>
To run the container use:

<pre><code>docker run -t --name test_container dp_simple_ga:first_draft</code></pre>

to start the container. Then:

<pre><code>docker exec -it test_container bash</code></pre>

to open a terminal in the container.

We also have included the docker file used to build this image.

It can be build using
<pre><code>docker build -t dp_simple_ga:artifact_v1 . 
</code></pre>
### Build From Source:
#### Dependencies:
We recommend using python 3.8
All dependencies are included in the requirements.txt file and can be installed with ```pip install -r requirements.txt```
- pandas==1.1.4
- scipy==1.5.4
- matplotlib==3.3.3
- numpy==1.19.4
- scikit_learn==0.24.1
- ray==2.1.0

Once the dependencies are installed, make sure to build the cython code using:
<pre><code>cd pam_algo && python3 setup.py install
</code></pre>
Finally, you will need to ```unzip processed_datasets.zip``` if you wish to use the exact datasets we used (or you can download and reprocess the datasets as described below)

## Running the Simple GA
To conduct a single training run of the simple GA you can run the ```our_dpga.py``` file directly.
Run ```python3 our_dpga.py --help``` for a complete list of arguments. All arguments have a default, so even ```python3 our_dpga.py``` works. The output is saved to a pickle file and some information is printed to the screen while running.


## Reproducing Experiments
We provide an example bash script for running the code run.sh.
The main.py file takes 3 primary args:

```--dataset_name``` which can take any value in [Adult, Credit, Mushrooms, Spam]

```--problem_name``` which can be LR or KM_* where * can be any number of medians (less than public set size) we use k=16

```--experiment``` with args descibed below:
| Experiment | Command |
|------------|---------|
|'generations'| Run the simple GA over various generations and epsilons|
|'pg_generations' | Run PrivGene over various generations and epsilons|
|'grid' | Run a grid search over all params of the simple GA|
|'non-priv_grid' | Run a grid search over all params of the simple GA - Non-private|
|'grid_rerun' | Run 100 runs of the best grid parameters|
|'privGene' | Run the default version of PrivGene|
|'jones' |Run Jones et al.'s $k$ median approach|
|'selection' | Test various selection and composition techniques on the simple GA|
|'utility' | Test the effect of distance matrix clipping|
|'privLocal' | Run the default version of PrivLocal|
|'privLocal_mod' | Run the modified version of PrivLocal|
|'privEEM' | Run the default version of PrivEEM|
|'privEEM_mod' | Run the modified version of PrivEEM|
|'gupta'| Run Gupta et al's Local Search algorithm|
|'gupta_mod'| Run our modified version of Gupta et al's Local Search algorithm|

We note to run all experiments takes over a week on a machine with 160CPUs. This would entail adding all experiments args above to the bash script. For simplicity, we leave only the smallest experiment, 'utility', which takes less than an hour on a 160CPU machine or perhaps a day on a quad-core. It can also be run on a single dataset and problem. For example, ```python3 main.py --experiment utility``` runs LR on Spam (the defaults) and takes only half an hour to run on a quad-core (default).

### Plotting
We included the jupiter notebook used to create the plots for our paper. We note that the experiments will need to be re-run for this script to work.

## Datasets
Datasets can be found here:
- Adult:  https://archive.ics.uci.edu/ml/datasets/adult
- Credit: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Spam: https://archive.ics.uci.edu/ml/datasets/spambase
- Mushrooms: https://archive.ics.uci.edu/ml/datasets/mushroom

Pre-processing is done by running ```dataset_maker.py```
Pickle files of the processed datasets are included in ```processed_datasets.zip```

## PrivGene Notes
PrivGene Code from https://sourceforge.net/projects/privgene/

Last Update: 2014-07-09

Initial Conversion to python by Lee and Kifer in: https://arxiv.org/pdf/1808.09501.pdf

Code: https://github.com/ppmlguy/DP-AGD

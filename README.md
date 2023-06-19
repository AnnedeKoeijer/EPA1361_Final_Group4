# Sensitivity Analysis Analysis


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Files](#files)
- [Usage](#usage)
- [Examples](#examples)

## Introduction

The sensitivity analysis provided in the code uses several techniques to evaluate the sensitivity of the results 
provided by the dike model. It uses different functions who each represent their own method of analysis to assess the 
impact of uncertainties on key performance indicators (KPIs), all methods come together in a final function. Each of 
them provides a certain results each saved in a corresponding directory, either in the form of a text file, csv file or 
png file. The defined KPI's are expected number of deaths, investment costs of building dikes, invest costs for the room 
for the river, expected annual damages and evacuation costs

## Installation

Besides the 3.11 version of python and the ema-workbench make sure the following packages are installed before trying to run the code: 
1. pandas
2. Numpy
3. matplotlib
4. seaborn
5. statsmodels
6. SALib
7. scikit-learn

## Files
Two other files that are part of the model were used in the sensitivity analysis
1. from dike_model_simulation: The function 'problem_formulation'
2. problem_formulation: The function 'get_model_for_problem_formulation'
They were used in order to receive the uncertainty parameters of the model


## Usage

The code consist out of several functions which are all called upon by one function: 
run_sensitivity_analysis(sobol, regular, name)
This requires you to enter the directory path of the sobol run, the directory path of th regular run and the name under 
which you would like the run to be saved. After running the analysis this folder can be found inside the 'results' 
folder. It will produce image files of the graphs produced by the convergence, sobol and tree analysis. 

## Examples
An example of the lines to actually run the code is provided by the following: 
    run_sensitivity_analysis("results/run_sobol_base.tar.gz", "results/run_base.tar.gz", 'Base_run_0_policies')
    run_sensitivity_analysis("results/run_sobol_policies.tar.gz", "results/run_policies.tar.gz", 'Base_run_10_policies')
Here the analysis is run for both a zero policies run and a 10 policies run, it is an exploratory analysis. 



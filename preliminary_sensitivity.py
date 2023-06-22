#!/usr/bin/env python
# coding: utf-8
#importing all libraries used for the analysis
from ema_workbench import load_results
import pandas as pd
import statsmodels.api as sm
import numpy as np
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ema_workbench.analysis import feature_scoring
from matplotlib.transforms import Bbox

#importing the needed functions from the exciting python scripts
from problem_formulation import get_model_for_problem_formulation
from dike_model_simulation import problem_formulation

#Setting the settings so that the results are presented correctly and are clear
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',50)

#getting the necesary information from the model settings
dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation)


def setup_dataframe_outcomes (outcomes):
    """
        Setup the dataframe for storing and manipulating outcomes.

        This function takes the outcomes dictionary with it's arrays as values and creates a dataframe to store the
        outcomes.It performs additional calculations that take into account the definition of time on which the
        variables are defined and transformations on the dataframe to aggregate and summarize the outcomes for each key
        performance indicator (KPI).

        Parameters:
        - outcomes (dict): Dictionary containing the outcomes of the model.

        Returns:
        - df_outcomes (pandas DataFrame): The dataframe containing the outcomes of the model that can be used for  analysis
        """
    # Splitting the arrays and creating a dataframe
    outcome = {f'{key} {i + 1}': value[:, i] for key, value in outcomes.items() for i in range(value.shape[1])}
    df_outcomes = pd.DataFrame(outcome)
    # Adding together columns representing one KPI and considering their time-based results
    for i in range(1,6):
        df_outcomes[f'A.{i}_Expected_Annual_Damage'] = (df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Expected Annual Damage')]].sum(axis=1))*(200/len(planning_steps))
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Expected Annual Damage')], axis=1)

        df_outcomes[f'A.{i}_Dike_Investment_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Dike Investment Costs')]].sum(axis=1)
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Dike Investment Costs')], axis=1)

        df_outcomes[f'A.{i}_Expected_Number_of_Deaths'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Expected Number of Deaths')]].sum(axis=1)*(200/len(planning_steps))
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Expected Number of Deaths')], axis=1)
    # Calculate total costs and drop the individual columns
    df_outcomes['RfR_Total_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith('RfR Total Costs')]].sum(axis=1)
    df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith('RfR Total Costs')], axis=1)
    # Calculate total evacuation costs and drop the individual columns
    df_outcomes['Expected_evacuation_costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith('Expected Evacuation Costs')]].sum(axis=1)
    df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith('Expected Evacuation Costs')], axis=1)
    # Calculate the defined kpi's
    df_outcomes['Total_Expected_Number_of_Deaths'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Expected_Number_of_Deaths')]].sum(axis=1)
    df_outcomes['Total_Dike_Investment_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Dike_Investment_Costs')]].sum(axis=1)
    df_outcomes['Total_Expected_Annual_Damage'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Expected_Annual_Damage')]].sum(axis=1)

    return df_outcomes

def run_sobol(df_outcomes_sobol, kpi, problem, name):
    """
    Perform Sobol' sensitivity analysis. This function calculates the Sobol' sensitivity indices
    for a given key performance indicator (KPI) and creates a plot to visualise the results

    Parameters:
        - df_outcomes (pandas DataFrame): DataFrame containing the outcomes of the model.
        - kpi (str): Key performance indicator to analyze.
        - problem (dict): The problem definition including the names of the uncertain variables and their
        corresponding ranges or probability distribution
        - name (str): The run name for the file in which to save the results

    Returns:
    None
    """
    # Extract the values of the specified KPI from the outcomes DataFrame
    final_sobol = df_outcomes_sobol[kpi].to_numpy()

    # Perform Sobol' sensitivity analysis
    Si = sobol.analyze(problem, final_sobol, calc_second_order=True, print_to_console=True)

    # Filter the sensitivity indices and create a DataFrame
    Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    Si_df = pd.DataFrame(Si_filter, index=problem['names'])

    try:
        # Set the plot style
        sns.set_style('white')

        # Create a new figure and axes
        fig, ax = plt.subplots(1)

        # Extract the S1 and ST indices and corresponding confidence intervals
        indices = Si_df[['S1', 'ST']]
        err = Si_df[['S1_conf', 'ST_conf']]

        # Create a bar plot of the sensitivity indices with error bars
        indices.plot.bar(yerr=err.values.T, ax=ax)

        # Set the figure size and adjust the bottom margin
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(bottom=0.3)

        # Create the directory if it doesn't exist
        os.makedirs(f'./results/{name}', exist_ok=True)

        # Save the plot to a file
        plt.savefig(f'./results/{name}/sobol_plot_{kpi}.png')

        # Close the figure to free up resources
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred while creating the plot: {str(e)}")
        # Perform any desired actions when the plot cannot be created
        # For example, return an empty plot, log the error, or continue with other code

def check_convergence_sobol (df_outcomes, kpi, problem, name):
    """
        Perform convergence analysis for Sobol' sensitivity indices. This function calculates the convergence
        of Sobol' indices for a given key performance indicator (KPI) and creates and saves a plot to
        visualise the results.

        Parameters:
        - df_outcomes (pandas DataFrame): DataFrame containing the outcomes of the model.
        - kpi (str): Key performance indicator to analyze.
        - problem (dict): The problem definition including the names of the uncertain variables and their
        corresponding ranges or probability distribution
        - name (str): The run name for the file in which to save the results

        Returns:
        None
        """

    # Extracting the values of the specified KPI from the outcomes DataFrame
    Y = df_outcomes[kpi].to_numpy()
    # Creating a DataFrame to store the sensitivity analysis results
    s_data = pd.DataFrame(index=problem['names'],
                          columns=np.arange(20,100,50)*(2*problem['num_vars']+2))
    # Perform sensitivity analysis for different sample sizes
    for j in s_data.columns:
        # Calculate Sobol' indices for the current sample size
        scores = sobol.analyze(problem, Y[0:j], calc_second_order=True, print_to_console=False)
        # Store the total indices (ST) in the DataFrame
        s_data.loc[:,j] = scores['ST']
    #Create the plot
    fig, ax = plt.subplots(1)
    #plot the convergence of Sobol' indices
    s_data.T.plot(ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Total index (ST)')
    ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5))
    # Define the bounding box for saving the plot
    bbox = Bbox.from_bounds(0, 0, 8, 4.7)
    # Save it to a png file
    plt.savefig(f'./results/{name}/scenario_check_convergence_{kpi}.png',dpi=300, bbox_inches=bbox)

def extra_trees(df_outcomes, experiments, name):
    """
        Perform Extra Trees analysis on the outcomes and uncertainties and save the results as heatmaps.

        Parameters:
        - df_outcomes (pandas DataFrame): DataFrame containing the outcomes of the model.
        - problem (dict): The problem definition including the names of the uncertain variables and their
        corresponding ranges or probability distribution
        - name (str): The run name for the file in which to save the results

        Returns:
        - None
    """
    # Extract the relevant outcomes for analysis
    df_outcomes_kpis = df_outcomes[['Total_Expected_Number_of_Deaths','Total_Dike_Investment_Costs',
            'Total_Expected_Annual_Damage', 'Expected_evacuation_costs', 'RfR_Total_Costs' ]].copy()
    # Perform Extra Trees analysis on model levers
    plt.figure(figsize=(15, 10))

    model_levers =[l.name for l in dike_model.levers]
    df_experiments_levers = experiments[model_levers]
    fs = feature_scoring.get_feature_scores_all(df_experiments_levers,df_outcomes_kpis)
    sns.heatmap(fs, cmap="viridis", annot=True)
    plt.savefig(f'./results/{name}/Tree_analysis_incl_policies.png', bbox_inches='tight')
    # Perform Extra Trees analysis on experiment uncertainties
    experiments_uncertainties = experiments[['A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail',
                                             'A.2_Bmax', 'A.2_Brate', 'A.2_pfail', 'A.3_Bmax', 'A.3_Brate', 'A.3_pfail',
                                             'A.4_Bmax', 'A.4_Brate', 'A.4_pfail', 'A.5_Bmax', 'A.5_Brate', 'A.5_pfail',
                                             'discount rate 0', 'discount rate 1', 'discount rate 2',
                                             'discount rate 3']].copy()

    fs = feature_scoring.get_feature_scores_all(experiments_uncertainties,df_outcomes_kpis)
    #Define the size of the figure
    plt.figure(figsize=(15, 10))
    #Plot the heat map
    sns.heatmap(fs, cmap="viridis", annot=True)
    #Save it to a png file
    plt.savefig(f'./results/{name}/Tree_analysis_uncertainties.png', bbox_inches='tight')


def check_convergence_extratrees(df_outcomes, experiments, kpi, name):
    """
        Perform convergence analysis using Extra Trees on the outcomes and uncertainties by gradually increasing the
        number of samples and calculating feature scores. The results are plotted and saved in a png file.

        Parameters:
        - df_outcomes (pandas DataFrame): DataFrame containing the outcomes of the model.
        - kpi (str): Key performance indicator to analyze.
        - df_experiments (pandas DataFrame): DataFrame containing the uncertainties and policies of the model
        - name (str): The run name for the file in which to save the results

        Returns:
        - None
    """
    # Select the experiment uncertainties for analysis
    experiments_uncertainties = experiments[[ 'A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail',
                                              'A.2_Bmax', 'A.2_Brate', 'A.2_pfail', 'A.3_Bmax', 'A.3_Brate',
                                              'A.3_pfail', 'A.4_Bmax', 'A.4_Brate', 'A.4_pfail', 'A.5_Bmax',
                                              'A.5_Brate', 'A.5_pfail', 'discount rate 0', 'discount rate 1',
                                              'discount rate 2', 'discount rate 3']]

    # Select the relevant outcomes for analysis
    df_outcomes_kpis = df_outcomes[
        ['Total_Expected_Number_of_Deaths', 'Total_Dike_Investment_Costs', 'Total_Expected_Annual_Damage',
         'Expected_evacuation_costs', 'RfR_Total_Costs']]

    # Create a DataFrame to store the convergence results
    s_data = pd.DataFrame(index=experiments_uncertainties.columns,
                          columns=np.arange(20, len(experiments), len(experiments)/20))

    # Perform convergence analysis by gradually increasing the number of samples
    for j in s_data.columns:
        j = int(j)
        scores = feature_scoring.get_feature_scores_all(experiments_uncertainties.iloc[:j, :],
                                                        df_outcomes_kpis.iloc[:j,:])
        s_data.loc[:, j] = scores[kpi]

    # Plot the convergence results
    fig, ax = plt.subplots(1)
    s_data.T.plot(ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Feature scores')

    # Adjust the legend and save the plot in the png file
    bbox = Bbox.from_bounds(0, 0, 9, 5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f'./results/{name}/scenario_check_convergence_extratrees_{kpi}.png',dpi=300, bbox_inches=bbox)

def run_sensitivity_analysis(sobol, regular, name):
    """
        Run the sensitivity analysis using various methods and save the results. This function combines multiple
        analysis methods to perform the entire sensitivity analysis. It creates necessary directories, loads the results,
        sets up dataframes, runs regressions, runs Sobol analysis, checks convergence using Sobol analysis, runs Extra
        Trees analysis, and checks convergence using Extra Trees analysis. The results are saved in the specified
        directory.

        Parameters:
        - sobol (str): File path to the results of the Sobol analysis.
        - regular (str): File path to the regular results.
        - name (str): Name under which the analysis results should be saved

        Returns:
        - None
    """
    # Create a directory for the analysis results if it doesn't exist
    if not os.path.exists(f'./results/{name}'):
         os.mkdir(f'./results/{name}')
    else:
        # Remove existing files in the directory
        for filename in os.listdir(f'./results/{name}'):
             file_path = os.path.join(f'./results/{name}', filename)
             if os.path.isfile(file_path):
                 os.remove(file_path)

    # Load the regular results
    experiments, outcomes = load_results(regular)

    # Load the Sobol analysis results
    experiments_sobol, outcomes_sobol = load_results(sobol)

    # Set up the outcomes dataframes for both the regular as the sobol analysis
    df_outcomes = setup_dataframe_outcomes(outcomes)
    df_outcomes_sobol = setup_dataframe_outcomes(outcomes_sobol)

    # Drop unnecessary columns from the experiments dataframes provided by the results
    df_experiments = experiments.drop(['policy', 'model', 'scenario'], axis = 1)
    df_experiments_sobol = experiments_sobol.drop(['policy', 'model', 'scenario'], axis = 1)


    # Get the SALib problem definition need to run te sobol analysis and check the convergence
    problem = get_SALib_problem(dike_model.uncertainties)

    # Run Sobol analysis for each KPI
    run_sobol (df_outcomes_sobol, 'Total_Expected_Number_of_Deaths', problem, name)
    run_sobol (df_outcomes_sobol, 'Total_Dike_Investment_Costs', problem, name)
    run_sobol (df_outcomes_sobol, 'Total_Expected_Annual_Damage', problem, name)
    run_sobol (df_outcomes_sobol, 'Expected_evacuation_costs', problem, name)
    run_sobol (df_outcomes_sobol, 'RfR_Total_Costs', problem, name)

    # Check convergence using Sobol analysis for each KPI
    check_convergence_sobol (df_outcomes_sobol, 'Total_Expected_Number_of_Deaths', problem, name)
    check_convergence_sobol (df_outcomes_sobol, 'Total_Dike_Investment_Costs', problem, name)
    check_convergence_sobol (df_outcomes_sobol, 'Total_Expected_Annual_Damage', problem, name)
    check_convergence_sobol (df_outcomes_sobol, 'Expected_evacuation_costs', problem, name)
    check_convergence_sobol (df_outcomes_sobol, 'RfR_Total_Costs', problem, name)

    # Run Extra Trees analysis
    extra_trees(df_outcomes, experiments, name)

    # Check convergence using Extra Trees analysis for each KPI
    check_convergence_extratrees(df_outcomes,experiments,'Total_Expected_Number_of_Deaths', name)
    check_convergence_extratrees(df_outcomes, experiments, 'Total_Expected_Annual_Damage', name)

# Perform sensitivity analysis for the base run with 0 policies and with 10 policies
run_sensitivity_analysis("results/run_sobol_base.tar.gz", "results/run_base.tar.gz", 'Base_run_0_policies')
run_sensitivity_analysis("results/run_sobol_policies.tar.gz", "results/run_policies.tar.gz", 'Base_run_10_policies')



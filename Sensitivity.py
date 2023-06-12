#!/usr/bin/env python
# coding: utf-8

from ema_workbench import load_results
import pandas as pd
import statsmodels.api as sm
import numpy as np
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from dike_model_simulation import problem_formulation
from SALib.analyze import sobol
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ema_workbench.analysis import feature_scoring
from matplotlib.transforms import Bbox
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',50)


dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation)





def setup_dataframe_outcomes (outcomes):
    outcome = {f'{key} {i + 1}': value[:, i] for key, value in outcomes.items() for i in range(value.shape[1])}
    df_outcomes = pd.DataFrame(outcome)
    for i in range(1,6):
        df_outcomes[f'A.{i}_Expected_Annual_Damage'] = (df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Expected Annual Damage')]].sum(axis=1))*(200/len(planning_steps))
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Expected Annual Damage')], axis=1)

        df_outcomes[f'A.{i}_Dike_Investment_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Dike Investment Costs')]].sum(axis=1)
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Dike Investment Costs')], axis=1)

        df_outcomes[f'A.{i}_Expected_Number_of_Deaths'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith(f'A.{i}_Expected Number of Deaths')]].sum(axis=1)*(200/len(planning_steps))
        df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith(f'A.{i}_Expected Number of Deaths')], axis=1)

    df_outcomes['RfR_Total_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith('RfR Total Costs')]].sum(axis=1)
    df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith('RfR Total Costs')], axis=1)

    df_outcomes['Expected_evacuation_costs'] = df_outcomes[[col for col in df_outcomes.columns if col.startswith('Expected Evacuation Costs')]].sum(axis=1)
    df_outcomes = df_outcomes.drop(df_outcomes.columns[df_outcomes.columns.str.startswith('Expected Evacuation Costs')], axis=1)

    df_outcomes['Total_Expected_Number_of_Deaths'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Expected_Number_of_Deaths')]].sum(axis=1)
    df_outcomes['Total_Dike_Investment_Costs'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Dike_Investment_Costs')]].sum(axis=1)
    df_outcomes['Total_Expected_Annual_Damage'] = df_outcomes[[col for col in df_outcomes.columns if col.endswith('Expected_Annual_Damage')]].sum(axis=1)

    return df_outcomes


def run_regression (kpi, df_outcomes, df_experiments, name):
    final_lhs = df_outcomes[kpi]
    X_0 = sm.add_constant(df_experiments)
    est = sm.OLS(final_lhs, X_0.astype(float)).fit()
    parameters_df = pd.DataFrame(est.params, columns=['Parameter'])
    parameters_df.index.name = 'Variable'
    # Save parameters to CSV
    parameters_df.to_csv(f'./results/{name}/parameters_{kpi}.csv')
    # Save summary to CSV
    with open(f'./results/{name}/summary_{kpi}.txt', 'w') as f:
        f.write(est.summary().as_text())


# #### 2. Sobol

def run_sobol(df_outcomes_sobol, kpi, problem, name):
    final_sobol = df_outcomes_sobol[kpi].to_numpy()
    Si = sobol.analyze(problem, final_sobol, calc_second_order=True, print_to_console=True)
    Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    Si_df = pd.DataFrame(Si_filter, index=problem['names'])

    try:
        sns.set_style('white')
        fig, ax = plt.subplots(1)

        indices = Si_df[['S1', 'ST']]
        err = Si_df[['S1_conf', 'ST_conf']]

        indices.plot.bar(yerr=err.values.T, ax=ax)
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(bottom=0.3)

        # Create the directory if it doesn't exist
        os.makedirs(f'./results/{name}', exist_ok=True)

        plt.savefig(f'./results/{name}/sobol_plot_{kpi}.png')

        plt.close(fig)  # Close the figure to free up resources
    except Exception as e:
        print(f"An error occurred while creating the plot: {str(e)}")
        # Perform any desired actions when the plot cannot be created
        # For example, return an empty plot, log the error, or continue with other code


def check_convergence_sobol (df_outcomes, kpi, problem, name, sort):
    Y = df_outcomes[kpi].to_numpy()
    s_data = pd.DataFrame(index=problem['names'],
                          columns=np.arange(20,100,50)*(2*problem['num_vars']+2))
    for j in s_data.columns:
        scores = sobol.analyze(problem, Y[0:j], calc_second_order=True, print_to_console=False)
        s_data.loc[:,j] = scores['ST']
    fig, ax = plt.subplots(1)

    s_data.T.plot(ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Total index (ST)')
    ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5))
    bbox = Bbox.from_bounds(0, 0, 8, 4.7)
    plt.savefig(f'./results/{name}/scenario_check_convergence_{kpi}_{sort}.png',dpi=300, bbox_inches=bbox)


# #### 3. Extra trees

def extra_trees(df_outcomes, experiments, name):
    df_outcomes_kpis = df_outcomes[['Total_Expected_Number_of_Deaths','Total_Dike_Investment_Costs',
            'Total_Expected_Annual_Damage', 'Expected_evacuation_costs', 'RfR_Total_Costs' ]].copy()
    plt.figure(figsize=(15, 10))

    model_levers =[l.name for l in dike_model.levers]
    df_experiments_levers = experiments[model_levers]

    fs = feature_scoring.get_feature_scores_all(df_experiments_levers,df_outcomes_kpis)
    sns.heatmap(fs, cmap="viridis", annot=True)
    plt.savefig(f'./results/{name}/Tree_analysis_incl_policies.png', bbox_inches='tight')

    experiments_uncertainties = experiments[['A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail', 'A.2_Bmax', 'A.2_Brate', 'A.2_pfail', 'A.3_Bmax', 'A.3_Brate', 'A.3_pfail', 'A.4_Bmax', 'A.4_Brate', 'A.4_pfail', 'A.5_Bmax', 'A.5_Brate', 'A.5_pfail', 'discount rate 0', 'discount rate 1', 'discount rate 2', 'discount rate 3']].copy()

    fs = feature_scoring.get_feature_scores_all(experiments_uncertainties,df_outcomes_kpis)
    plt.figure(figsize=(15, 10))
    sns.heatmap(fs, cmap="viridis", annot=True)
    plt.savefig(f'./results/{name}/Tree_analysis_uncertainties.png', bbox_inches='tight')


def check_convergence_extratrees(df_outcomes, experiments, kpi, name):
    experiments_uncertainties = experiments[[
        'A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_Brate', 'A.1_pfail', 'A.2_Bmax', 'A.2_Brate',
        'A.2_pfail', 'A.3_Bmax', 'A.3_Brate', 'A.3_pfail', 'A.4_Bmax', 'A.4_Brate', 'A.4_pfail',
        'A.5_Bmax', 'A.5_Brate', 'A.5_pfail', 'discount rate 0', 'discount rate 1', 'discount rate 2',
        'discount rate 3']]

    df_outcomes_kpis = df_outcomes[
        ['Total_Expected_Number_of_Deaths', 'Total_Dike_Investment_Costs', 'Total_Expected_Annual_Damage',
         'Expected_evacuation_costs', 'RfR_Total_Costs']]

    s_data = pd.DataFrame(index=experiments_uncertainties.columns,
                          columns=np.arange(20, len(experiments), len(experiments)/20))

    for j in s_data.columns:
        j = int(j)
        scores = feature_scoring.get_feature_scores_all(experiments_uncertainties.iloc[:j, :],
                                                        df_outcomes_kpis.iloc[:j,:])
        s_data.loc[:, j] = scores[kpi]

    fig, ax = plt.subplots(1)
    s_data.T.plot(ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Feature scores')

    bbox = Bbox.from_bounds(0, 0, 9, 5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f'./results/{name}/scenario_check_convergence_extratrees_{kpi}.png',dpi=300, bbox_inches=bbox)

# #### 4. Running the full code

def run_sensitivity_analysis(sobol, regular, name):
    if not os.path.exists(f'./results/{name}'):
         os.mkdir(f'./results/{name}')
    else:
         for filename in os.listdir(f'./results/{name}'):
             file_path = os.path.join(f'./results/{name}', filename)
             if os.path.isfile(file_path):
                 os.remove(file_path)
        # shutil.rmtree(f'./results/{name}')
        # os.mkdir(f'./results/{name}')
    
    experiments, outcomes = load_results(regular)
    experiments_sobol, outcomes_sobol = load_results(sobol)
    df_outcomes = setup_dataframe_outcomes(outcomes)
    df_outcomes_sobol = setup_dataframe_outcomes(outcomes_sobol)

    df_experiments = experiments.drop(['policy', 'model', 'scenario'], axis = 1)
    df_experiments_sobol = experiments_sobol.drop(['policy', 'model', 'scenario'], axis = 1)
    #run_regression ('Total_Expected_Number_of_Deaths', df_outcomes, df_experiments, name)
    #run_regression ('Total_Dike_Investment_Costs', df_outcomes, df_experiments, name)
    #run_regression ('Total_Expected_Annual_Damage', df_outcomes, df_experiments, name)
    #run_regression ('Expected_evacuation_costs', df_outcomes, df_experiments, name)
    #run_regression ('RfR_Total_Costs', df_outcomes, df_experiments, name)

    problem = get_SALib_problem(dike_model.uncertainties)

    run_sobol (df_outcomes_sobol, 'Total_Expected_Number_of_Deaths', problem, name)
    run_sobol (df_outcomes_sobol, 'Total_Dike_Investment_Costs', problem, name)
    run_sobol (df_outcomes_sobol, 'Total_Expected_Annual_Damage', problem, name)
    run_sobol (df_outcomes_sobol, 'Expected_evacuation_costs', problem, name)
    run_sobol (df_outcomes_sobol, 'RfR_Total_Costs', problem, name)

    check_convergence_sobol (df_outcomes_sobol, 'Total_Expected_Number_of_Deaths', problem, name, 'sobol')
    check_convergence_sobol (df_outcomes_sobol, 'Total_Dike_Investment_Costs', problem, name, 'sobol')
    check_convergence_sobol (df_outcomes_sobol, 'Total_Expected_Annual_Damage', problem, name, 'sobol')
    check_convergence_sobol (df_outcomes_sobol, 'Expected_evacuation_costs', problem, name, 'sobol')
    check_convergence_sobol (df_outcomes_sobol, 'RfR_Total_Costs', problem, name, 'sobol')


    extra_trees(df_outcomes, experiments, name)

    check_convergence_extratrees(df_outcomes,experiments,'Total_Expected_Number_of_Deaths', name)
    check_convergence_extratrees(df_outcomes, experiments, 'Total_Expected_Annual_Damage', name)


run_sensitivity_analysis("results/run_sobol_base.tar.gz", "results/run_base.tar.gz", 'Base_run_0_policies')
run_sensitivity_analysis("results/run_sobol_policies.tar.gz", "results/run_policies.tar.gz", 'Base_run_10_policies')

print("done")



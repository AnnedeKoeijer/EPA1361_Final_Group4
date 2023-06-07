{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true,
    "ExecuteTime": {
     "end_time": "2023-06-06T13:44:27.461000400Z",
     "start_time": "2023-06-06T13:44:27.448388800Z"
    }
   },
   "outputs": [],
   "source": [
    "from ema_workbench import load_results\n",
    "import pandas as pd\n",
    "from ema_workbench.analysis import prim\n",
    "from ema_workbench import ema_logging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "outputs": [],
   "source": [
    "experiments, outcomes = load_results('./results/run_base.tar.gz')"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2023-06-06T13:44:28.285574900Z",
     "start_time": "2023-06-06T13:44:28.253781300Z"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "outputs": [],
   "source": [
    "# df_outcomes = pd.DataFrame({key: np.concatenate(value) for key, value in outcomes.items()})\n",
    "outcome = {f'{key} {i + 1}': value[:, i] for key, value in outcomes.items() for i in range(value.shape[1])}\n",
    "df_outcomes = pd.DataFrame(outcome)\n",
    "df_outcomes.to_excel('results/firstcheck.xlsx')\n",
    "\n",
    "for i in range(1,6):\n",
    "    df_outcomes[f'A.{i}_Expected_Anual_Dagame'] = df_outcomes[f'A.{i}_Expected Annual Damage 1'] + df_outcomes[f'A.{i}_Expected Annual Damage 2'] + df_outcomes[f'A.{i}_Expected Annual Damage 3']\n",
    "    df_outcomes = df_outcomes.drop([f'A.{i}_Expected Annual Damage 1', f'A.{i}_Expected Annual Damage 2', f'A.{i}_Expected Annual Damage 3'], axis = 1)\n",
    "\n",
    "    df_outcomes[f'A.{i}_Dike_Investment_Costs'] = df_outcomes[f'A.{i}_Dike Investment Costs 1'] + df_outcomes[f'A.{i}_Dike Investment Costs 2'] + df_outcomes[f'A.{i}_Dike Investment Costs 3']\n",
    "    df_outcomes = df_outcomes.drop([f'A.{i}_Dike Investment Costs 1', f'A.{i}_Dike Investment Costs 2', f'A.{i}_Dike Investment Costs 3'], axis = 1)\n",
    "\n",
    "    df_outcomes[f'A.{i}_Expected_Number_of_Deaths'] = df_outcomes[f'A.{i}_Expected Number of Deaths 1'] + df_outcomes[f'A.{i}_Expected Number of Deaths 2'] + df_outcomes[f'A.{i}_Expected Number of Deaths 3']\n",
    "    df_outcomes = df_outcomes.drop([f'A.{i}_Expected Number of Deaths 1', f'A.{i}_Expected Number of Deaths 2', f'A.{i}_Expected Number of Deaths 3'], axis = 1)\n",
    "\n",
    "df_outcomes.to_excel('results/check.xlsx')\n"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2023-06-06T13:44:41.636140600Z",
     "start_time": "2023-06-06T13:44:41.163957100Z"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Expected Annual Damage 1', 'Expected Annual Damage 2',\n",
      "       'Expected Annual Damage 3', 'Expected Annual Damage 4',\n",
      "       'Dike Investment Costs 1', 'Dike Investment Costs 2',\n",
      "       'Dike Investment Costs 3', 'Dike Investment Costs 4',\n",
      "       'Expected Number of Deaths 1', 'Expected Number of Deaths 2',\n",
      "       'Expected Number of Deaths 3', 'Expected Number of Deaths 4',\n",
      "       'RfR Total Costs 1', 'RfR Total Costs 2', 'RfR Total Costs 3',\n",
      "       'RfR Total Costs 4', 'Expected Evacuation Costs 1',\n",
      "       'Expected Evacuation Costs 2', 'Expected Evacuation Costs 3',\n",
      "       'Expected Evacuation Costs 4'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "experiments.columns"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2023-06-06T13:46:22.705775100Z",
     "start_time": "2023-06-06T13:46:22.696822500Z"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "outputs": [
    {
     "ename": "AssertionError",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mAssertionError\u001B[0m                            Traceback (most recent call last)",
      "Cell \u001B[1;32mIn[50], line 4\u001B[0m\n\u001B[0;32m      2\u001B[0m experiments \u001B[38;5;241m=\u001B[39m experiments\u001B[38;5;241m.\u001B[39mastype(\u001B[38;5;28mint\u001B[39m)\n\u001B[0;32m      3\u001B[0m df_outcomes \u001B[38;5;241m=\u001B[39m df_outcomes\u001B[38;5;241m.\u001B[39mastype(\u001B[38;5;28mint\u001B[39m)\n\u001B[1;32m----> 4\u001B[0m prim_alg \u001B[38;5;241m=\u001B[39m \u001B[43mprim\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mPrim\u001B[49m\u001B[43m(\u001B[49m\u001B[43mdf_outcomes\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mexperiments\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mthreshold\u001B[49m\u001B[38;5;241;43m=\u001B[39;49m\u001B[38;5;241;43m0.8\u001B[39;49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mpeel_alpha\u001B[49m\u001B[38;5;241;43m=\u001B[39;49m\u001B[38;5;241;43m0.1\u001B[39;49m\u001B[43m)\u001B[49m\n\u001B[0;32m      5\u001B[0m box1 \u001B[38;5;241m=\u001B[39m prim_alg\u001B[38;5;241m.\u001B[39mfind_box()\n",
      "File \u001B[1;32m~\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ema_workbench\\analysis\\prim.py:979\u001B[0m, in \u001B[0;36mPrim.__init__\u001B[1;34m(self, x, y, threshold, obj_function, peel_alpha, paste_alpha, mass_min, threshold_type, mode, update_function)\u001B[0m\n\u001B[0;32m    965\u001B[0m \u001B[38;5;28;01mdef\u001B[39;00m \u001B[38;5;21m__init__\u001B[39m(\n\u001B[0;32m    966\u001B[0m     \u001B[38;5;28mself\u001B[39m,\n\u001B[0;32m    967\u001B[0m     x,\n\u001B[1;32m   (...)\u001B[0m\n\u001B[0;32m    976\u001B[0m     update_function\u001B[38;5;241m=\u001B[39m\u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mdefault\u001B[39m\u001B[38;5;124m\"\u001B[39m,\n\u001B[0;32m    977\u001B[0m ):\n\u001B[0;32m    978\u001B[0m     \u001B[38;5;28;01massert\u001B[39;00m mode \u001B[38;5;129;01min\u001B[39;00m {sdutil\u001B[38;5;241m.\u001B[39mRuleInductionType\u001B[38;5;241m.\u001B[39mBINARY, sdutil\u001B[38;5;241m.\u001B[39mRuleInductionType\u001B[38;5;241m.\u001B[39mREGRESSION}\n\u001B[1;32m--> 979\u001B[0m     \u001B[38;5;28;01massert\u001B[39;00m \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39m_assert_mode(y, mode, update_function)\n\u001B[0;32m    980\u001B[0m     \u001B[38;5;66;03m# preprocess x\u001B[39;00m\n\u001B[0;32m    981\u001B[0m     x \u001B[38;5;241m=\u001B[39m x\u001B[38;5;241m.\u001B[39mcopy()\n",
      "\u001B[1;31mAssertionError\u001B[0m: "
     ]
    }
   ],
   "source": [
    "experiments = experiments.drop(['policy','model'], axis = 1)\n",
    "experiments = experiments.astype(int)\n",
    "df_outcomes = df_outcomes.astype(int)\n",
    "prim_alg = prim.Prim(df_outcomes, experiments, threshold=0.8, peel_alpha=0.1)\n",
    "box1 = prim_alg.find_box()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime": {
     "end_time": "2023-06-05T12:14:39.503646100Z",
     "start_time": "2023-06-05T12:14:39.465399900Z"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [],
   "metadata": {
    "collapsed": false
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

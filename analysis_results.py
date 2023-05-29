import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import feature_scoring

fn = "./data/results/run_1.tar.gz"
experiments, outcomes = load_results(fn)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import feature_scoring

fn = "./results/run_base.tar.gz"
experiments, outcomes = load_results(fn)
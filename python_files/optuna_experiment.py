import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

import torch
from torch import nn


def print_intermediate_values(study, trial):
    '''Prints information about the current run once the run is completed'''
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print("Trial completed: Trial number {}".format(trial.number))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print("  Value: {}".format(trial.value))
    else:
        print("Trial {} failed.".format(trial.number))

def plot_optuna_results(val_losses, val_accuracies, figsize=(15,5), exp_to_include=''):
  '''Plots the optuna results.  Takes validation loss and validation accuries as inputs
  to create loss and accuracy curves.  Option to limit to a set list of experiments from
  the dataframe.'''
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize) 
  
  if exp_to_include == '':
    exp_to_include = list(range(len(val_losses)))

  for i, data in enumerate(val_losses):
    if i in exp_to_include:
      exp_num = 'Experiment_' + str(i)
      ax1.plot(data, label=exp_num)
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.set_title('Validation Loss vs Epoch')
  ax1.legend()

  for i, data in enumerate(val_accuracies):
    if i in exp_to_include:
      exp_num = 'Experiment_' + str(i)
      ax2.plot(data, label=exp_num)
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Loss')
  ax2.set_title('Validation Accuracy vs Epoch')
  ax2.legend()

  plt.show()


def save_optuna_results(study, train_losses_total, val_losses_total, val_accuracies_total, file_path, file_name):
    '''Save the optuna results to a dataframe for later use'''
    df = study.trials_dataframe()
    df_plots = pd.DataFrame({
        'train_losses': train_losses_total,
        'validation_losses': val_losses_total,
        'validation_accuracies': val_accuracies_total
    })

    # Save study results
    optuna_file_folder = 'data/optuna_results/'
    optuna_file_path = file_path + optuna_file_folder + file_name + '.csv'
    df.to_csv(optuna_file_path, index=False)

    # Save training results
    optuna_plots_file_path = file_path + optuna_file_folder + file_name + '.pkl'
    df_plots.to_pickle(optuna_plots_file_path)

    return df, df_plots


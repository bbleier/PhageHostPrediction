import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns


def plot_loss_and_accuracy(df, model_type='protein_only', single_or_double='single', include_other='False', figsize=(15,5), normalize=False, title=''):
    '''Pass in a dataframe that contains training losses, validation losses, and validation ac curacies to plot the loss and accuracy plots'''

    # Select the proper subset of data
    df = df[(df['model_type'] == model_type) & (df['single_or_double'] == single_or_double) & (df['include_other'] == include_other)].copy()

    # Normalize the loss data
    def normalize_loss(x):
        initial = x[0]
        return [i / initial for i in x]
    df['train_losses_norm'] = df['train_losses'].apply(normalize_loss)
    df['validation_losses_norm'] = df['validation_losses'].apply(normalize_loss)

    # Collect the data
    if normalize:
        train_losses = df['train_losses_norm'].tolist()
        val_losses = df['validation_losses_norm'].tolist()
    else:
        train_losses = df['train_losses'].tolist()
        val_losses = df['validation_losses'].tolist()
    val_accuracies= df['validation_accuracies'].tolist()

    # Properly label the number of hosts
    num_hosts = [2,5,10,20]
    num_hosts = num_hosts[0:len(df)]

    # Set up plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize) 

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, .8, len(num_hosts))]

    for i in range(len(df)):
        ax1.plot(train_losses[i], linestyle='-', linewidth=2, color=colors[i])
        ax1.plot(val_losses[i], linestyle='--', linewidth=2, color=colors[i])
        ax2.plot(val_accuracies[i], label=str(num_hosts[i]) + ' Hosts', linestyle='-', linewidth=2, color=colors[i])

    # Set up Loss figure
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.tick_params(axis='x', labelsize=13)
    ax1.tick_params(axis='y', labelsize=13)
    if title == '':
        ax1.set_title('Loss vs Epoch', fontsize=17)
    else: 
        ax1.set_title(title + ' Loss', fontsize=17)
    if normalize:
        ax1.set_ylim([0.7, 1])
    else:
        ax1.set_ylim([0, 4])

    # Create custom legend
    train_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='Training Loss')
    val_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Validation Loss')
    ax1.legend(handles=[train_line, val_line], loc='upper right', title="Legend", fontsize=11)

    # Set up accuracy figure
    ax2.set_xlabel('Epochs', fontsize=15)
    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.tick_params(axis='x', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax2.set_ylim([0, 1.05])
    if title == '':
        ax2.set_title('Accuracy vs Epoch', fontsize=17)
    else: 
        ax2.set_title(title + ' Accuracy', fontsize=17)
    ax2.legend(fontsize=11)

    plt.show()


def plot_single_vs_dual_head(df, figsize=(7,5)):
  '''Pass in a dataframe that contains accuriacies for single headed and dual
  headed model.  Will plot comparison figure between model types with seaborn'''

  df_single_no_other = df[(df['model_type'] == 'protein_only') & (df['single_or_double'] == 'single') & (df['include_other'] == 'False')]
  df_single_other = df[(df['model_type'] == 'protein_only') & (df['single_or_double'] == 'single') & (df['include_other'] == 'True')]
  df_double_other = df[(df['model_type'] == 'protein_only') & (df['single_or_double'] == 'double') & (df['include_other'] == 'True')]

  # Extract final accuracy value
  final_accuracy_single_no_other = []
  final_accuracy_single_other = []
  final_accuracy_double_other = []

  for i in range(len(df_single_no_other)):
    final_accuracy_single_no_other.append(df_single_no_other.iloc[i]['validation_accuracies'][-1])
    final_accuracy_single_other.append(df_single_other.iloc[i]['validation_accuracies'][-1])
    final_accuracy_double_other.append(df_double_other.iloc[i]['validation_accuracies'][-1])

  print(final_accuracy_single_no_other)
  print(final_accuracy_single_other)
  print(final_accuracy_double_other)

  # Properly label the number of hosts
  num_hosts = ['2 Hosts','5 Hosts','10 Hosts','20 Hosts']

  # Set up plot data for seaborn
  plot_data = pd.DataFrame({
      'Number of Hosts': num_hosts * 3,  # Repeat num_hosts for each category
      'Final Accuracy': final_accuracy_single_no_other + final_accuracy_single_other + final_accuracy_double_other,
      'Model Type': ['Single-Headed (no "Other" Class)'] * len(final_accuracy_single_no_other) + ['Single-Headed (w/ "Other" Class)'] * len(final_accuracy_single_other) + ['Dual-Headed (w/ Other Class)'] * len(final_accuracy_double_other)
  })

  # Plot with Seaborn
  plt.figure(figsize=figsize)
  sns.barplot(data=plot_data, x='Number of Hosts', y='Final Accuracy', hue='Model Type', palette='viridis')

  plt.title('Accuracy for Single vs. Dual-Headed Models', fontsize=15)
  plt.ylabel('Final Accuracy', fontsize=13)
  plt.xlabel('')
  plt.xticks(fontsize=13)
  plt.ylim(0,1.15)
  plt.legend()
  plt.show()


def plot_data_distribution_pie_chart(df, figsize=(10,8)):
    '''Pass in a dataframe containing labels of top n-classes.   Will plot a pie 
    chart of the data distribution among those classes'''

    # Setting up data
    temp_df = df.groupby('adjusted_host_name')['refseq_id'].count().sort_values(ascending=False)
    temp_df = pd.DataFrame(temp_df)
    temp_df['percentage'] = (temp_df['refseq_id'] / temp_df['refseq_id'].sum()) * 100
    temp_df.reset_index(inplace=True)

    # Set up color scheme
    colors = plt.cm.viridis(np.linspace(0.25, 1, len(temp_df)))

    # Create 'explode' array to separate the slices a bit
    explode = [0.05] * len(temp_df)  # Slightly explode each slice

    # Create figure
    plt.figure(figsize=figsize)
    plt.pie(temp_df['percentage'], labels=temp_df['adjusted_host_name'], autopct='%1.1f%%', startangle=70, colors=colors, explode = explode, labeldistance=1.05, textprops={'fontsize': 13})
    plt.title('Distribution of Data by Host Name', fontsize=18)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
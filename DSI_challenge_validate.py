#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set
Date: 13JUN2021
Author: John Koenig
Purpose: Conduct analysis and predicitive modeling on the given breast cancer
         data set

Inputs: breast-cancer.csv

Outputs: breast_cancer_output.csv
         trained_model
Notes: 
    
'''


#%%
# import libraries

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns;

import pdb

import warnings
warnings.filterwarnings("ignore")  # supress DataConversionWarning

#project_dir = os.getcwd()
project_dir = r'/home/md33a/python_projects/DSI_challenge/'
data_dir = project_dir + r'data/'
output_dir = project_dir + r'output/'
plots_dir = output_dir + r'plots/'

# set dpi
dpi = 300

# save files or not
save_files = True


#%%
# Load Random Forest Train/Test Metrics DataFrame

# load data
df_metrics_filename = 'train_test_metrics.csv'
df_metrics = pd.read_csv(output_dir + df_metrics_filename)
df_metrics_head = df_metrics.iloc[:100,:]
df_metrics_columns = list(df_metrics.columns)

# random forest hyperparameter options
max_sample_percent_options = [s / 100 for s in range(5, 105, 5)]
max_depth_options = list(range(1,21))

# prepare metrics
test_metrics_list = ['true_positive_rate',
                     'true_negative_rate',
                     'false_positive_rate',
                     'false_negative_rate',
                     'accuracy',
                     'precision',
                     'recall',
                     'F1_score',
                     'ROC_area']

metric_columns = [str(d) for d in max_depth_options]
metric_index = [f'{str(int(i * 100))}%' for i in max_sample_percent_options]

#%%
# Reshape Metrics Data

# filter for only test records
test_df_metrics = df_metrics[df_metrics['train_test'] == 'Test']
test_df_metrics.sort_values(['max_sample_percent', 'max_depth'], inplace=True)
test_df_metrics.index = range(len(test_df_metrics))

# create metric dataframes
true_positive_rate_df = pd.DataFrame(columns=metric_columns)
true_negative_rate_df = pd.DataFrame(columns=metric_columns)
false_positive_rate_df = pd.DataFrame(columns=metric_columns)
false_negative_rate_df = pd.DataFrame(columns=metric_columns)
accuracy_df = pd.DataFrame(columns=metric_columns)
precision_df = pd.DataFrame(columns=metric_columns)
recall_df = pd.DataFrame(columns=metric_columns)
F1_score_df = pd.DataFrame(columns=metric_columns)
ROC_area_df = pd.DataFrame(columns=metric_columns)

# build individual metrics dataframes
for sample in max_sample_percent_options:
    
    sample_df_tmp = test_df_metrics[test_df_metrics['max_sample_percent'] == sample]
    
    true_positive_rate_tmp = pd.DataFrame([sample_df_tmp['true_positive_rate'].values], columns=metric_columns)
    true_positive_rate_df = pd.concat([true_positive_rate_df, true_positive_rate_tmp], axis=0)
    
    true_negative_rate_tmp = pd.DataFrame([sample_df_tmp['true_negative_rate'].values], columns=metric_columns)
    true_negative_rate_df = pd.concat([true_negative_rate_df, true_negative_rate_tmp], axis=0)
    
    false_positive_rate_tmp = pd.DataFrame([sample_df_tmp['false_positive_rate'].values], columns=metric_columns)
    false_positive_rate_df = pd.concat([false_positive_rate_df, false_positive_rate_tmp], axis=0)
    
    false_negative_rate_tmp = pd.DataFrame([sample_df_tmp['false_negative_rate'].values], columns=metric_columns)
    false_negative_rate_df = pd.concat([false_negative_rate_df, false_negative_rate_tmp], axis=0)
    
    accuracy_tmp = pd.DataFrame([sample_df_tmp['accuracy'].values], columns=metric_columns)
    accuracy_df = pd.concat([accuracy_df, accuracy_tmp], axis=0)
    
    precision_tmp = pd.DataFrame([sample_df_tmp['precision'].values], columns=metric_columns)
    precision_df = pd.concat([precision_df, precision_tmp], axis=0)
    
    recall_tmp = pd.DataFrame([sample_df_tmp['recall'].values], columns=metric_columns)
    recall_df = pd.concat([recall_df, recall_tmp], axis=0)
    
    F1_score_tmp = pd.DataFrame([sample_df_tmp['F1_score'].values], columns=metric_columns)
    F1_score_df = pd.concat([F1_score_df, F1_score_tmp], axis=0)
    
    ROC_area_tmp = pd.DataFrame([sample_df_tmp['ROC_area'].values], columns=metric_columns)
    ROC_area_df = pd.concat([ROC_area_df, ROC_area_tmp], axis=0)

true_positive_rate_df.index = metric_index
true_negative_rate_df.index = metric_index
false_positive_rate_df.index = metric_index
false_negative_rate_df.index = metric_index
accuracy_df.index = metric_index
precision_df.index = metric_index
recall_df.index = metric_index
F1_score_df.index = metric_index
ROC_area_df.index = metric_index

if save_files:
    true_positive_rate_df.to_csv(output_dir + 'true_positive_rate_df.csv')
    true_negative_rate_df.to_csv(output_dir + 'true_negative_rate_df.csv')
    false_positive_rate_df.to_csv(output_dir + 'false_positive_rate_df.csv')
    false_negative_rate_df.to_csv(output_dir + 'false_negative_rate_df.csv')
    accuracy_df.to_csv(output_dir + 'accuracy_df.csv')
    precision_df.to_csv(output_dir + 'precision_df.csv')
    recall_df.to_csv(output_dir + 'recall_df.csv')
    F1_score_df.to_csv(output_dir + 'F1_score_df.csv')
    ROC_area_df.to_csv(output_dir + 'ROC_area_df.csv')


#%%
# Create Heatmap Multiplots

x_ticks = [0, 5, 10, 15, 20]
x_tick_labels = ['1', '5', '10', '15', '20']
y_ticks = [0, 5, 10, 15, 20]
y_tick_labels = ['0%', '25%', '50%', '75%', '100%']
x_axis_label = 'Max Tree Depth'
y_axis_label = 'Bootstrap Sample Percentage'

y_tick_labels.reverse()

fig, ax = plt.subplots(2, 4, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle('Test Performance Heatmaps', x=.525, fontsize=42)

sns.heatmap(np.flip(true_positive_rate_df.values, 0), cmap='Greens', ax=ax[0,0])
sns.heatmap(np.flip(true_negative_rate_df.values, 0), cmap='Reds_r', ax=ax[0,1])
sns.heatmap(np.flip(accuracy_df.values, 0), cmap='Oranges', ax=ax[0,2])
sns.heatmap(np.flip(precision_df.values, 0), cmap='Purples', ax=ax[0,3])
sns.heatmap(np.flip(false_positive_rate_df.values, 0), cmap='Reds_r', ax=ax[1,0])
sns.heatmap(np.flip(false_negative_rate_df.values, 0), cmap='Greens', ax=ax[1,1])
sns.heatmap(np.flip(recall_df.values, 0), cmap='Blues', ax=ax[1,2])
sns.heatmap(np.flip(F1_score_df.values, 0), cmap='Greys', ax=ax[1,3])

title_fontsize = 16
tick_fontsize = 14
cbar_kws = {'fontsize':tick_fontsize}

ax[0,0].set_title('true_positive_rate', fontsize=title_fontsize)
ax[0,1].set_title('true_negative_rate', fontsize=title_fontsize)
ax[0,2].set_title('accuracy', fontsize=title_fontsize)
ax[0,3].set_title('precision', fontsize=title_fontsize)
ax[1,0].set_title('false_positive_rate', fontsize=title_fontsize)
ax[1,1].set_title('false_negative_rate', fontsize=title_fontsize)
ax[1,2].set_title('recall', fontsize=title_fontsize)
ax[1,3].set_title('F1_score', fontsize=title_fontsize)

ax[1,0].set_xticks(x_ticks)
ax[1,0].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)

ax[1,1].set_xticks(x_ticks)
ax[1,1].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)

ax[1,2].set_xticks(x_ticks)
ax[1,2].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)

ax[1,3].set_xticks(x_ticks)
ax[1,3].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)

ax[0,0].set_yticks(y_ticks)
ax[0,0].set_yticklabels(y_tick_labels, fontsize=tick_fontsize, rotation=90)

ax[1,0].set_yticks(y_ticks)
ax[1,0].set_yticklabels(y_tick_labels, fontsize=tick_fontsize, rotation=90)

fig.text(.44, .07, x_axis_label, fontsize=24)
fig.text(.09, .31, y_axis_label, fontsize=24, rotation=90)
plt.subplots_adjust(left=.13, top=.89)

if save_files:
    
    # save heatmap plot
    confusion_heatmaps_filename = 'confusion_heatmaps.png'
    fig.savefig(plots_dir + confusion_heatmaps_filename, dpi=dpi)


#%%
# Select Tree and Visualize

test_df_metrics_sort_order = ['false_negative_rate',
                              'true_positive_rate',
                              'true_negative_rate',
                              'false_positive_rate']

test_df_metrics_ascending_order = [True, False, False, True] 

test_df_metrics.sort_values(by=test_df_metrics_sort_order,
                            axis=0,
                            ascending=test_df_metrics_ascending_order,
                            inplace=True)

# extract optimal metrics
lowest_fnr = test_df_metrics['false_negative_rate'].min()
highest_tpr = test_df_metrics['true_positive_rate'].max()

# select potential models
potential_models = test_df_metrics[test_df_metrics['false_negative_rate'] == lowest_fnr]
potential_models = potential_models[potential_models['true_positive_rate'] == highest_tpr]
  
potential_models.index = range(len(potential_models))

# create potential models table
max_table_rows = 10
potential_models10 = potential_models.iloc[:max_table_rows,:]

table_columns = [#'model_name',
                 #'train_test',
                 'test_split',
                 'total_records',
                 'n_trees',
                 'max_sample_percent',
                 'max_depth',
                 'true_positive',
                 'true_negative',
                 'false_positive',
                 'false_negative',
                 'true_positive_rate',
                 'true_negative_rate',
                 'false_positive_rate',
                 'false_negative_rate',
                 'accuracy',
                 'precision',
                 'recall',
                 'F1_score',
                 'ROC_area']

potential_models10 = potential_models10[table_columns]
potential_models10['false_positive_rate'] = potential_models10['false_positive_rate'].round(4)
potential_models10_table_strings = [list(r) for _,r in potential_models10.astype(str).iterrows()]

table_fontsize = 9
column_widths = [.1, .15, .1, .2, .1, .15, .15, .15, .15, .2, .2, .2, .2, .1, .1, .1, .1, .1]
row_labels = potential_models['model_name'].iloc[:max_table_rows]

fig, ax = plt.subplots(figsize=(10,4))
fig.suptitle('Potential Random Forest Models', x=1, fontsize=28)
ax.set_axis_off()

model_table = ax.table(potential_models10_table_strings,
                       colWidths = column_widths,
                       rowColours =['lightgrey'] * len(potential_models10.columns),
                       rowLabels = row_labels,
                       colColours =['lightblue'] * len(potential_models10.columns),
                       colLabels = table_columns,
                       cellLoc ='center',
                       loc ='upper left')

model_table.scale(1,2)

model_table.auto_set_font_size(False)
model_table.set_fontsize(table_fontsize)

if save_files:
    
    # save potential_models dataframe
    potential_models_filename = 'potential_models.csv'
    potential_models.to_csv(output_dir + potential_models_filename)
    
    # save potential models table
    potential_models_filename = 'potenital_models_table.png'
    #fig.savefig(plots_dir + potential_models_filename)        # this isn't working properly

    
         

            
            
    
    
    
    













































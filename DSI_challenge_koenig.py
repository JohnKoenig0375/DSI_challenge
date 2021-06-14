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
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns;

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics


warnings.filterwarnings("ignore")  # supress DataConversionWarning

#project_dir = os.getcwd()
project_dir = r'/home/md33a/python_projects/DSI_challenge/'
data_dir = project_dir + r'data/'
output_dir = project_dir + r'output/'
models_dir = output_dir + r'models/'
plots_dir = output_dir + r'plots/'
ROC_train_dir = plots_dir + r'train/'
ROC_test_dir = plots_dir + r'test/'

# set dpi
dpi = 300

# save files or not
save_files = True

#%%
# Download and Clean Project DataFrame

# download data
url = r'https://gist.githubusercontent.com/jeff-boykin/b5c536467c30d66ab97cd1f5c9a3497d/raw/5233c792af49c9b78f20c35d5cd729e1307a7df7/breast-cancer.csv'
df = pd.read_csv(url, header=None)

# create column labels
input_variables_labels = ['var' + str(i) for i in range(len(df.columns) - 2)]
column_labels = ['ID', 'target'] + input_variables_labels
df.columns = column_labels

# separate data
target = df['target'].str.contains('M').astype(int)
ID_list = df['ID']
input_variables = df.iloc[:,2:]

# calculate % of dataset that is positive class for target variable
percent_target = np.round(target.sum() / target.size, 2)

if save_files:
    
    # save cleaned data
    df_filename = 'breast-cancer.csv'
    df.to_csv(data_dir + df_filename, index=False)


#%%
# Create DataFrame to Store Train/Test Results

metrics_df_columns = ['model_name',
                      'train_test',
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

metrics_df = pd.DataFrame(columns=metrics_df_columns)

metrics_df_filename = 'train_test_metrics.csv'

if save_files and os.path.exists(metrics_df_filename) == False:
    
    # create metrics dataframe
    metrics_df.to_csv(output_dir + metrics_df_filename, index=False)


#%%
# Correlation Heatmap

# create mask for upper right of triangle
mask = np.triu(np.ones_like(input_variables.corr(), dtype=bool))

# create correlation heatmap for input variables
corr_min = -1
corr_max = 1

fig, ax = plt.subplots(figsize=(9.5, 8))

sns.heatmap(input_variables.corr(),
            vmin=corr_min,
            vmax=corr_max,
            cmap='coolwarm_r',
            cbar_kws={"ticks": [-1, 0, 1]},
            mask=mask,
            ax=ax)

fig.suptitle('Correlation Between Breast Cancer Input Variables', x=.46, fontsize=18)

plt.subplots_adjust(top=.92)


if save_files:
    
    # save heatmap plot
    corr_filename = 'corr_heatmap.png'
    fig.savefig(plots_dir + corr_filename, dpi=dpi)


#%%
# Small Multiples of Histograms

# create a histogram for each variable
data_min = input_variables.min().sort_values(ascending=True).values[0]
data_max = input_variables.max().sort_values(ascending=False).values[0]
data_range = (data_min, data_max)

axe_per_fig = 6

bins = 100
x_limit = (0, 5000)
y_limit = (0, 50)
hist_color = 'mediumblue'

# iterate over input variables and create custom histograms for each
for p in range(int(len(input_variables_labels) / axe_per_fig)):
    
    start_index = p * 6
    end_index = p * 6 + 6
    var_labels_tmp = input_variables_labels[start_index:end_index]
    
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    
    ax_list = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2]]
    
    for i in range(len(ax_list)):    
        ax_list[i].hist(input_variables[var_labels_tmp[i]], bins=bins, color=hist_color)
        ax_list[i].set_title(var_labels_tmp[i])
        ax_list[i].set_ylim(y_limit)
    
    suptitle = f'Histograms of Breast Cancer Input Variables {var_labels_tmp[0]} - {var_labels_tmp[5]}'
    suptitle += '\nNote: X-axis is not constant between plots'
    fig.suptitle(suptitle, x=.55, fontsize=18)
    fig.subplots_adjust(top=0.87)  # make room for fig title
    
    # add axis labels
    x_axis_label = f'Values [bins={bins}]'
    y_axis_label = f'Value Counts'
    fig.text(.44, .05, x_axis_label, fontsize=14)
    fig.text(.08, .45, y_axis_label, fontsize=14, rotation=90)
    
    # add vline for mean median
    lines_labels = ['Mean', 'Median']
    colors = ['limegreen', 'tomato']
    line_styles = ['--', '--']
    alpha = .8
    
    for i in range(len(ax_list)):
        mean_tmp = input_variables[var_labels_tmp[i]].mean()
        med_tmp = input_variables[var_labels_tmp[i]].median()
        
        ax_list[i].axvline(x=mean_tmp, color=colors[0], linestyle=line_styles[0], alpha=alpha)
        ax_list[i].axvline(x=med_tmp, color=colors[1], linestyle=line_styles[1], alpha=alpha)
        
    # create custom legend
    handles = [mlines.Line2D([], [], color=colors[0], linestyle=line_styles[0], alpha=alpha),
               mlines.Line2D([], [], color=colors[1], linestyle=line_styles[1], alpha=alpha)]
    
    fig.legend(handles, lines_labels, loc='right', fontsize=12)
    fig.subplots_adjust(right=.88)  # make room for custom legend
    
    if save_files:
        
        # save histogram multi-plot
        histogram_6plots_filename = f'histogram_6plots_part{p}.png'
        fig.savefig(plots_dir + histogram_6plots_filename, dpi=dpi)


#%%
# 30 Variable Box Plot

# reshape input data
input_variables_labels_reversed = input_variables_labels[::-1]
box_input = input_variables[input_variables_labels_reversed].T

# create box plots
fig, ax = plt.subplots(figsize=(6, 12))

ax.boxplot(box_input, vert=False)

ax.set_yticklabels(input_variables_labels_reversed)
ax.set_title('Breast Cancer Input Variables - Box Plot', x=.45)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Variable Name', fontsize=12)

if save_files:
    
    # save box plot
    boxplots30_filename = f'boxplots30.png'
    fig.savefig(plots_dir + boxplots30_filename, dpi=dpi)


#%%
# 30 Variable Box Plot - Outliers Removed

# remove outliers
outlier_cutoff = 1
no_outliers_list = []

for v in input_variables_labels:
    if input_variables[v].max() <= outlier_cutoff:
        no_outliers_list.append(v)

input_variables_no_outliers = input_variables[no_outliers_list]

# reshape input data
input_variables_no_outliers_labels_reversed = input_variables_no_outliers.columns.to_list()[::-1]
box_input = input_variables[input_variables_no_outliers_labels_reversed].T

# create box plots
fig, ax = plt.subplots(figsize=(6, 12))

ax.boxplot(box_input, vert=False)

ax.set_yticklabels(input_variables_no_outliers_labels_reversed)
ax.set_title('Breast Cancer Input Variables - Box Plot\nNo Outliers', x=.44)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Variable Name', fontsize=12)

if save_files:
    
    # save plot - no outliers
    boxplots_no_outliers_filename = f'boxplots_no_outliers.png'
    fig.savefig(plots_dir + boxplots_no_outliers_filename, dpi=dpi)


#%%
# Scale and Split Dataset for Training 75-25

# scale input variables
scaler = MinMaxScaler()
input_fit = scaler.fit(input_variables)
input_scaled = input_fit.transform(input_variables)

# split dataset into train/test partitions
x = pd.concat([ID_list, pd.DataFrame(input_scaled)], axis=1)
y = target.values

test_split = .25
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, stratify=y)

y_train = pd.DataFrame(y_train, columns=['y_train'])
y_test = pd.DataFrame(y_test, columns=['y_test'])

x_train.index = range(len(x_train))
x_test.index = range(len(x_test))
y_train.index = range(len(y_train))
y_test.index = range(len(y_test))

# save IDs
train_IDs = x_train.iloc[:,0]
test_IDs = x_test.iloc[:,0]

# drop ID column
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

if save_files:
    
    # save Scaler model
    scaler_filename = f'scaler_model.pkl'
    pickle.dump(input_fit, open(models_dir + scaler_filename, 'wb'))
    
    # save scaled data
    df_scaled = pd.concat([ID_list, target, pd.DataFrame(input_scaled)], axis=1)
    
    df_scaled_filename = 'df_scaled.csv'
    df_scaled.to_csv(output_dir + df_scaled_filename, index=False)
    
    # save data splits
    pd.concat([train_IDs, x_train, y_train], axis=1).to_csv(output_dir + 'train_split.csv')
    pd.concat([test_IDs, x_test, y_test], axis=1).to_csv(output_dir + 'test_split.csv')


#%%
# Conduct Scaling and Principal Component Analysis

# conduct PCA
n_components = 20

pca = PCA(n_components=n_components)

pca_fit = pca.fit(x_train)
pca_variance = np.round(pca_fit.explained_variance_ratio_, decimals=3)*100
pca_variance_cum = np.cumsum(np.round(pca_fit.explained_variance_ratio_, decimals=3)*100)

input_pca = pca_fit.transform(x_train)

# get eigenvalues
vars_covar = np.cov(x_train)
eigen_values, eigen_vectors = np.linalg.eig(vars_covar)
eigen_values = np.round(eigen_values.astype(float), 2)[:n_components]

# select final components
final_components = 3
x_train_pca = input_pca[:,:final_components]

if save_files:
    
    # save PCA model
    pca_final_filename = f'PCA_final_components{final_components}.pkl'
    pickle.dump(pca_fit, open(models_dir + pca_final_filename, 'wb'))
    
    # save PCA output
    df_pca = pd.concat([train_IDs, y_train, pd.DataFrame(x_train_pca)], axis=1)
    
    df_pca_filename = f'df_pca_components{final_components}.csv'
    df_pca.to_csv(output_dir + df_pca_filename, index=False)


#%%
# Train Random Forest Model

max_sample_percent_options = [s / 100 for s in range(5, 101, 5)]
max_depth_options = list(range(1,21))

n_trees = 10000
decimals = 4

for sample in max_sample_percent_options:
    for depth in max_depth_options:

        # train random forest
        max_sample_percent = sample
        max_depth = depth
        max_samples = int(len(x_train_pca) * max_sample_percent)
        
        model_name = f'RF_trees{n_trees}_sample{sample}_depth{max_depth}'
        train_test = 'Train'
        
        #print(f'Training Model - {model_name}')
        
        x = x_train_pca
        y = y_train
        
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    max_depth=max_depth,
                                    bootstrap=True,
                                    max_samples=max_samples)
        
        rf_model = rf.fit(x, y)
        rf_train_predict = rf.predict(x)
        
        # get confusion matrix and metrics for train partition
        confusion_matrix_train = metrics.confusion_matrix(rf_train_predict, y_train)
        
        total_records = len(x_train_pca)
        
        true_positive = confusion_matrix_train[0,0]
        true_negative = confusion_matrix_train[1,1]
        false_positive = confusion_matrix_train[0,1]
        false_negative = confusion_matrix_train[1,0]
        
        true_positive_rate = np.round(true_positive / total_records, decimals)
        true_negative_rate = np.round(true_negative / total_records, decimals)
        false_positive_rate = np.round(false_positive / total_records, decimals)
        false_negative_rate = np.round(false_negative / total_records, decimals)
        
        accuracy = np.round((true_positive + false_negative) / total_records, decimals)
        precision = np.round(true_positive / (true_positive + false_positive), decimals)
        recall = np.round(true_positive / (true_positive + false_negative), decimals)
        F1_score = np.round(2 / ((1 / precision) + (1 / recall)), decimals)
        
        ROC_area = np.round(metrics.roc_auc_score(y_train, rf_train_predict), decimals)
        
        # plot ROC curve
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', ls='--')
        
        metrics.plot_roc_curve(rf_model,
                               x_train_pca,
                               y_train,
                               ax=ax)
        
        ROC_string = f'ROC Curve -- {train_test} -- {model_name}'
        ax.set_title(ROC_string)
        
        if save_files:
            
            # save ROC curve plot
            ROC_plot_filename = f'{ROC_string}_{train_test}.png'
            fig.savefig(ROC_train_dir + ROC_plot_filename, dpi=dpi)
        
            # save model train metrics
            metrics_series = pd.DataFrame([[model_name,
                                            train_test,
                                            test_split,
                                            total_records,
                                            n_trees,
                                            max_sample_percent,
                                            max_depth,
                                            true_positive,
                                            true_negative,
                                            false_positive,
                                            false_negative,
                                            true_positive_rate,
                                            true_negative_rate,
                                            false_positive_rate,
                                            false_negative_rate,
                                            accuracy,
                                            precision,
                                            recall,
                                            F1_score,
                                            ROC_area]], columns=metrics_df_columns)
            
            metrics_df = pd.concat([metrics_df, metrics_series], axis=0)
            metrics_df.index = range(len(metrics_df))
            
            # save metrics in 'train_test_metrics.csv'
            metrics_df.to_csv(output_dir + metrics_df_filename,
                              header=False,
                              mode='a',
                              index=False)
            
            # clear metrics
            metrics_df = pd.DataFrame(columns=metrics_df_columns)
            
            # save random forest model
            rf_model_filename = f'{model_name}.pkl'
            pickle.dump(rf_model, open(models_dir + rf_model_filename, 'wb'))
        
        
        #%%
        # Test Random Forest Model
        
        train_test = 'Test'
        
        print(f'Testing Model - {model_name}')
        
        # apply PCA model
        x_test_pca = pca_fit.transform(x_test)[:,:final_components]
        
        # save PCA test data
        df_pca = pd.concat([test_IDs, y_test, pd.DataFrame(x_test_pca)], axis=1)
        
        df_pca_filename = f'df_pca_components{final_components}.csv'
        df_pca.to_csv(output_dir + df_pca_filename, index=False)
        
        # get predictions
        x = x_test_pca
        y = y_test.values
        
        rf_test_predict = rf_model.predict(x)
        
        # get confusion matrix and metrics for test partition
        confusion_matrix_train = metrics.confusion_matrix(rf_test_predict, y_test)
        
        total_records = len(x_test_pca)
        
        true_positive = confusion_matrix_train[0,0]
        true_negative = confusion_matrix_train[1,1]
        false_positive = confusion_matrix_train[0,1]
        false_negative = confusion_matrix_train[1,0]
        
        true_positive_rate = np.round(true_positive / total_records, decimals)
        true_negative_rate = np.round(true_negative / total_records, decimals)
        false_positive_rate = np.round(false_positive / total_records, decimals)
        false_negative_rate = np.round(false_negative / total_records, decimals)
        
        accuracy = np.round((true_positive + false_negative) / total_records, decimals)
        precision = np.round(true_positive / (true_positive + false_positive), decimals)
        recall = np.round(true_positive / (true_positive + false_negative), decimals)
        F1_score = np.round(2 / ((1 / precision) + (1 / recall)), decimals)
        
        ROC_area = np.round(metrics.roc_auc_score(y_test, rf_test_predict), decimals)
        
        # plot ROC curve
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', ls='--')
        
        metrics.plot_roc_curve(rf_model,
                               x_test_pca,
                               y_test,
                               ax=ax)
        
        ROC_string = f'ROC Curve -- {train_test} -- {model_name}'
        ax.set_title(ROC_string)
        
        if save_files:
            
            # save ROC curve plot
            ROC_plot_filename = f'{ROC_string}.png'
            fig.savefig(ROC_test_dir + ROC_plot_filename, dpi=dpi)
            
            # save model test metrics
            metrics_series = pd.DataFrame([[model_name,
                                            train_test,
                                            test_split,
                                            total_records,
                                            n_trees,
                                            max_sample_percent,
                                            max_depth,
                                            true_positive,
                                            true_negative,
                                            false_positive,
                                            false_negative,
                                            true_positive_rate,
                                            true_negative_rate,
                                            false_positive_rate,
                                            false_negative_rate,
                                            accuracy,
                                            precision,
                                            recall,
                                            F1_score,
                                            ROC_area]], columns=metrics_df_columns)
            
            metrics_df = pd.concat([metrics_df, metrics_series], axis=0)
            metrics_df.index = range(len(metrics_df))
            
            # save metrics in 'train_test_metrics.csv'
            metrics_df.to_csv(output_dir + metrics_df_filename,
                              header=False,
                              mode='a',
                              index=False)
            
            # clear metrics
            metrics_df = pd.DataFrame(columns=metrics_df_columns)


















































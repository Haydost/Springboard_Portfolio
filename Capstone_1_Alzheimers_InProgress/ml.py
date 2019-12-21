#!/usr/bin/env python3
"""Machine Learning module for ADNI capstone project.

This module contains functions for use with the ADNI dataset.
"""

if 'pd' not in globals():
    import pandas as pd

if 'np' not in globals():
    import numpy as np
    
if 'plt' not in globals():
    import matplotlib.pyplot as plt
    
if 'sns' not in globals():
    import seaborn as sns
    
if 'scipy.stats' not in globals():
    import scipy.stats
    
if 'StandardScaler' not in globals():
    from sklearn.preprocessing import StandardScaler

if 'KNeighborsClassifier' not in globals():
    from sklearn.neighbors import KNeighborsClassifier


sns.set()


def get_delta_scaled(final_exam, neg_one=False):
    """Take the final_exam dataframe and return datasets.
    
    This function returns five numpy arrays: feature_names, X_delta_male, 
    X_delta_female, y_delta_male, and y_delta_female. The two X arrays hold
    the feature data. The two y arrays hold the diagnosis group labels.
    The feature_names array hold a list of the features. The neg_one
    parameter allows you to specify -1 for the negative class (for SVM)."""
    
    # map the diagnosis group and assign to dx_group
    nc_idx = final_exam[final_exam.DX == final_exam.DX_bl2].index
    cn_mci_idx = final_exam[(final_exam.DX == 'MCI') & (final_exam.DX_bl2 == 'CN')].index
    mci_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'MCI')].index
    cn_ad_idx = final_exam[(final_exam.DX == 'AD') & (final_exam.DX_bl2 == 'CN')].index

    if neg_one:
        labels = pd.concat([pd.DataFrame({'dx_group': -1}, index=nc_idx),
                            pd.DataFrame({'dx_group': -1}, index=cn_mci_idx),
                            pd.DataFrame({'dx_group': 1}, index=mci_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=cn_ad_idx)
                           ]).sort_index()
    else:
        labels = pd.concat([pd.DataFrame({'dx_group': 0}, index=nc_idx),
                            pd.DataFrame({'dx_group': 0}, index=cn_mci_idx),
                            pd.DataFrame({'dx_group': 1}, index=mci_ad_idx),
                            pd.DataFrame({'dx_group': 1}, index=cn_ad_idx)
                           ]).sort_index()
    
    # add to the dataframe and ensure every row has a label
    deltas_df = final_exam.loc[labels.index]
    deltas_df.loc[:,'dx_group'] = labels.dx_group 

    # convert gender to numeric column
    deltas_df = pd.get_dummies(deltas_df, drop_first=True, columns=['PTGENDER'])
    
    # extract the features for change in diagnosis
    X_delta = deltas_df.reindex(columns=['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta',
                                         'RAVLT_delta', 'Hippocampus_delta', 'Ventricles_delta',
                                         'WholeBrain_delta', 'Entorhinal_delta', 'MidTemp_delta',
                                         'PTGENDER_Male'])
      
    # store the feature names
    feature_names = np.array(['CDRSB_delta', 'ADAS11_delta', 'ADAS13_delta', 'MMSE_delta', 'RAVLT_delta',
                              'Hippocampus_delta', 'Ventricles_delta', 'WholeBrain_delta',
                              'Entorhinal_delta', 'MidTemp_delta', 'PTGENDER_Male'])
    
    # standardize the data
    scaler = StandardScaler()
    Xd = scaler.fit_transform(X_delta)
    
    # extract the labels
    yd = np.array(deltas_df.dx_group)
    
    # return the data
    return feature_names, Xd, yd

def plot_best_k(X_train, X_test, y_train, y_test, kmax=9):
    """This function will create a plot to help choose the best k for k-NN.
    
    Supply the training and test data to compare accuracy at different k values.
    Specifying a max k value is optional."""
    
    # Setup arrays to store train and test accuracies
    # view the plot to help pick the best k to use
    neighbors = np.arange(1, kmax)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
    
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)
    
    if kmax < 11:
        s = 2
    elif kmax < 21:
        s = 4
    elif kmax < 41:
        s = 5
    elif kmax < 101:
        s = 10
    else:
        s = 20
        
    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,kmax,s))
    plt.show()
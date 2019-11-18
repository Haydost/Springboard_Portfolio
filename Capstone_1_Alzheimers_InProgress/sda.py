#!/usr/bin/env python3
"""Statistical data analysis module for Alzheimer's Capstone 1.

This module contains functions used to extract data for and to perform
statistical analysis on the ADNI Alzheimer's Disease dataset for my
Capstone project. Inputs for these functions can be obtained using 
adnidatawrangling module, and some additional wrangling from the eda module.
Required modules for all functions to run: pandas, numpy, matplotlib.pyplot,
seaborn all using standard namespaces.
"""

if 'pd' not in globals():
    import pandas as pd

if 'np' not in globals():
    import numpy as np
    
if 'plt' not in globals():
    import matplotlib.pyplot as plt
    
if 'sns' not in globals():
    import seaborn as sns
    
sns.set()

def get_deltadx_groups():
    """This function uses the final_exam dataframe to divide the data by diagnosis change.
    
    The groups returned by this function are no_change, cn_mci, mci_ad, and cn_ad.
    """
    
    # isolate patients with no diagnosis change
    no_change = final_exam[final_exam['DX'] == final_exam['DX_bl2']]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = final_exam[(final_exam['DX'] == 'MCI') & (final_exam['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = final_exam[(final_exam['DX'] == 'AD') & (final_exam['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = final_exam[(final_exam['DX'] == 'AD') & (final_exam['DX_bl2'] == 'CN')]
    
    return no_change, cn_mci, mci_ad, cn_ad

def divide_genders(df):
    """This function divides the supplied dataframe by the 'PTGENDER' column.
    
    Returns two dataframes: males, females.
    """
    
    males = df[df.PTGENDER == 'Male']
    females = df[df.PTGENDER == 'Female']
    
    return males, females

def test_gender_deltas(df, biomarker):
    """This function returns the p value for the test that males/females have the 
    
    same distribution for the change in the supplied biomarker. A significant p value
    means that males/females should be separated for further analysis of change in 
    in the supplied biomarker. A high p value means males/females can be considered 
    together.
    """
    
    # get all of the data in a numpy array
    comb_arr = np.array(df[biomarker])
    
    # calculate the mean of the entire array
    comb_mean = np.mean(comb_arr)
    
    # initialize empty arrays for mean calculations
    null_means1 = np.empty(10000)
    null_means2 = np.empty(10000)
    
    for i in range(10000):
        # mix the values in a random order
        rand_arr = np.random.permutation(comb_arr)
        
        # create two arrays the same size as the male and female arrays
        null_arr1 = rand_arr[:len(df[df.PTGENDER == 'Male'])]
        null_arr2 = rand_arr[len(df[df.PTGENDER == 'Female']):]
        
        # get means and add to array
        null_means1[i] = np.mean(null_arr1)
        null_means2[i] = np.mean(null_arr2)
    
    # calculate and display p value
    p = np.sum(comb_mean >= abs(np.mean(null_means1) - np.mean(null_means2))) / len(comb_arr)
    print('Distribution Test for Males/Females')
    print('Variable: ', biomarker)
    print('If p < 0.05, then split the data by gender')
    print('p-value: ', p)
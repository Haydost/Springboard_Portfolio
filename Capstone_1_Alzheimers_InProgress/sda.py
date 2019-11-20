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

def get_deltadx_groups(df):
    """This function uses the supplied dataframe to divide the data by diagnosis change.
    
    The groups returned by this function are no_change, cn_mci, mci_ad, and cn_ad.
    """
    
    # isolate patients with no diagnosis change
    no_change = df[df['DX'] == df['DX_bl2']]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = df[(df['DX'] == 'MCI') & (df['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = df[(df['DX'] == 'AD') & (df['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = df[(df['DX'] == 'AD') & (df['DX_bl2'] == 'CN')]
    
    return no_change, cn_mci, mci_ad, cn_ad

def divide_genders(df):
    """This function divides the supplied dataframe by the 'PTGENDER' column.
    
    Returns two dataframes: males, females.
    """
    
    males = df[df.PTGENDER == 'Male']
    females = df[df.PTGENDER == 'Female']
    
    return males, females

def test_gender_effect(df, biomarker, size):
    """This function returns the p value for the test that males/females have the 
    
    same distribution for the change in the supplied biomarker. The test will be 
    performed over 'size' permutations. A significant p value means that males/females 
    should be separated for further analysis of change in in the supplied biomarker. 
    A high p value means males/females can be considered together.
    """
    
    # create a combined array for the biomarker
    c_arr = np.array(df[biomarker])
    
    # divide the data by gender
    fe_males = df[df.PTGENDER == 'Male']
    fe_females = df[df.PTGENDER == 'Female']

    # get counts of the number of males and females
    num_males = df.PTGENDER.value_counts()['Male']
    
    # calculate the observed mean difference
    obs_mean_diff = np.mean(fe_males[biomarker]) - np.mean(fe_females[biomarker])
    
    # initialize empty numpy array
    perm_mean_diffs = np.empty(size)
    
    # run the permutations calculating means each time
    for i in range(size):
        r_arr = np.random.permutation(c_arr)
        null_arr1 = r_arr[:num_males]
        null_arr2 = r_arr[num_males:]
        perm_mean_diffs[i] = np.mean(null_arr1) - np.mean(null_arr2)
    
    # uncomment to quickly view the distribution
    _ = plt.hist(perm_mean_diffs, density=True, color='blue', label='Perms')
    _ = plt.axvline(obs_mean_diff, color='C1')
    _ = plt.title('Probability Distribution for Mean Differences\nBetween Genders for ' + biomarker)
    _ = plt.xlabel('Mean Difference Between Males/Females')
    _ = plt.ylabel('Probability Density')
    
    # calculate and display p value
    if obs_mean_diff > np.mean(perm_mean_diffs):
        p = np.sum(perm_mean_diffs >= obs_mean_diff) / len(perm_mean_diffs)
    else:
        p = np.sum(perm_mean_diffs <= obs_mean_diff) / len(perm_mean_diffs)
    print('Distribution Test for Males/Females')
    print('Variable: ', biomarker)
    print('If p < 0.05, then split the data by gender')
    print('p-value: ', p)
    
def bs(df, biomarker, size):
    """This function generates and plots a bootstrap distribution.
    
    Supply the dataframe, biomarker, and number of samples to take for each distribution.
    This function returns the 95% confidence interval and plots the distribution.
    """
    
    # create the bootstrap distribution
    bs_df = pd.DataFrame({biomarker: [np.mean(np.random.choice(df[biomarker], 
                                                                size=len(df))) for i in range(size)]})
    
    # calculate and display the 95% confidence interval
    lower = bs_df[biomarker].quantile(0.025)
    upper = bs_df[biomarker].quantile(0.975)
    print('95% Confidence Interval: ', lower, ' to ', upper)
    
    # create and display histogram of the bootstrap distribution
    _ = bs_df[biomarker].hist(histtype='step')
    _ = plt.axvline(lower, color='C1', linewidth=1)
    _ = plt.axvline(upper, color='C1', linewidth=1)
    _ = plt.title('Bootstrap Estimate around the Mean for ' + biomarker + '\nNo Diagnosis Change')
    _ = plt.xlabel('Resampled Mean ' + biomarker)
    _ = plt.ylabel('Frequency')
    
    if abs(upper) > abs(lower):
        return upper
    else:
        return lower
    
def old_eval_bs(fe, biomarker, conf, gender='both'):
    """Old function. Code unsuccint. Created new version following DRY.
    
    Saving this old code because never delete anything that works.
    Calculate percentages of patients with a change in diagnosis that had
    a change in the biomarker larger than the threshold value identified from
    bootstrap analysis. You must supply the full final_exam dataframe, the biomarker 
    of interest, the confidence level to evaluate, and provide optional gender of male/female.
    """
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = fe[(fe['DX'] == 'MCI') & (fe['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'CN')]
    
    if gender == 'both':
        if conf > 0:
            end_CN = len(fe[(fe['DX'] == 'CN') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])
            end_MCI = len(fe[(fe['DX'] == 'MCI') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])
            end_AD = len(fe[(fe['DX'] == 'AD') & (fe[biomarker] > conf)]) / len(fe[fe[biomarker] > conf])     
            prog_CN_MCI = len(cn_mci[cn_mci[biomarker] > conf]) / len(cn_mci)
            prog_MCI_AD = len(mci_ad[mci_ad[biomarker] > conf]) / len(mci_ad)
            prog_CN_AD = len(cn_ad[cn_ad[biomarker] > conf]) / len(cn_ad)
        else:
            end_CN = len(fe[(fe['DX'] == 'CN') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            end_MCI = len(fe[(fe['DX'] == 'MCI') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            end_AD = len(fe[(fe['DX'] == 'AD') & (fe[biomarker] < conf)]) / len(fe[fe[biomarker] < conf])
            prog_CN_MCI = len(cn_mci[cn_mci[biomarker] < conf]) / len(cn_mci)
            prog_MCI_AD = len(mci_ad[mci_ad[biomarker] < conf]) / len(mci_ad)
            prog_CN_AD = len(cn_ad[cn_ad[biomarker] < conf]) / len(cn_ad)
    elif gender == 'males':
        m_cn_mci = cn_mci[cn_mci.PTGENDER == 'Male']
        m_mci_ad = mci_ad[mci_ad.PTGENDER == 'Male']
        m_cn_ad = cn_ad[cn_ad.PTGENDER == 'Male']
        m = fe[fe['PTGENDER'] == 'Male']
        if conf > 0:
            end_CN = len(m[(m['DX'] == 'CN') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])
            end_MCI = len(m[(m['DX'] == 'MCI') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])
            end_AD = len(m[(m['DX'] == 'AD') & (m[biomarker] > conf)]) / len(m[m[biomarker] > conf])     
            prog_CN_MCI = len(m_cn_mci[m_cn_mci[biomarker] > conf]) / len(m_cn_mci)
            prog_MCI_AD = len(m_mci_ad[m_mci_ad[biomarker] > conf]) / len(m_mci_ad)
            prog_CN_AD = len(m_cn_ad[m_cn_ad[biomarker] > conf]) / len(m_cn_ad)
        else:
            end_CN = len(m[(m['DX'] == 'CN') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf])
            end_MCI = len(m[(m['DX'] == 'MCI') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf])
            end_AD = len(m[(m['DX'] == 'AD') & (m[biomarker] < conf)]) / len(m[m[biomarker] < conf]) 
            prog_CN_MCI = len(m_cn_mci[m_cn_mci[biomarker] < conf]) / len(m_cn_mci)
            prog_MCI_AD = len(m_mci_ad[m_mci_ad[biomarker] < conf]) / len(m_mci_ad)
            prog_CN_AD = len(m_cn_ad[m_cn_ad[biomarker] < conf]) / len(m_cn_ad)
    else:
        f_cn_mci = cn_mci[cn_mci.PTGENDER == 'Female']
        f_mci_ad = mci_ad[mci_ad.PTGENDER == 'Female']
        f_cn_ad = cn_ad[cn_ad.PTGENDER == 'Female']
        f = fe[fe['PTGENDER'] == 'Female']
        if conf > 0:
            end_CN = len(f[(f['DX'] == 'CN') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            end_MCI = len(f[(f['DX'] == 'MCI') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            end_AD = len(f[(f['DX'] == 'AD') & (f[biomarker] > conf)]) / len(f[f[biomarker] > conf])
            prog_CN_MCI = len(f_cn_mci[f_cn_mci[biomarker] > conf]) / len(f_cn_mci)
            prog_MCI_AD = len(f_mci_ad[f_mci_ad[biomarker] > conf]) / len(f_mci_ad)
            prog_CN_AD = len(f_cn_ad[f_cn_ad[biomarker] > conf]) / len(f_cn_ad)
        else:
            end_CN = len(f[(f['DX'] == 'CN') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            end_MCI = len(f[(f['DX'] == 'MCI') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            end_AD = len(f[(f['DX'] == 'AD') & (f[biomarker] < conf)]) / len(f[f[biomarker] < conf])
            prog_CN_MCI = len(f_cn_mci[f_cn_mci[biomarker] < conf]) / len(f_cn_mci)
            prog_MCI_AD = len(f_mci_ad[f_mci_ad[biomarker] < conf]) / len(f_mci_ad)
            prog_CN_AD = len(f_cn_ad[f_cn_ad[biomarker] < conf]) / len(f_cn_ad)

    # print results
    print('Percent exceeding threshold that ended CN: ', round(end_CN*100,2), '%')
    print('Percent exceeding threshold that ended MCI: ', round(end_MCI*100,2), '%')
    print('Percent exceeding threshold that ended AD: ', round(end_AD*100,2), '%')
    print('Percent progressing CN to MCI exceeding threshold: ', round(prog_CN_MCI*100,2), '%')
    print('Percent Progressing MCI to AD exceeding threshold: ', round(prog_MCI_AD*100,2), '%')
    print('Percent Progressing CN to AD exceeding threshold: ', round(prog_CN_AD*100,2), '%')
    
def eval_bs(fe, biomarker, conf, gender='both'):
    """Calculate percentages of patients with a change in diagnosis that had
    
    a change in the biomarker larger than the threshold value identified from
    bootstrap analysis. You must supply the full final_exam dataframe, the biomarker 
    of interest, the confidence level to evaluate, and provide optional gender of male/female.
    """
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_mci = fe[(fe['DX'] == 'MCI') & (fe['DX_bl2'] == 'CN')]
    
    # isolate patients who progressed from 'MCI' to 'AD'
    mci_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'MCI')]
    
    # isolate patients who progressed from 'CN' to 'AD'
    cn_ad = fe[(fe['DX'] == 'AD') & (fe['DX_bl2'] == 'CN')]
    
    if gender == 'both':
        df = fe
        df2 = cn_mci
        df3 = mci_ad
        df4 = cn_ad
    elif gender == 'males':
        df = fe[fe['PTGENDER'] == 'Male']
        df2 = cn_mci[cn_mci.PTGENDER == 'Male']
        df3 = mci_ad[mci_ad.PTGENDER == 'Male']
        df4 = cn_ad[cn_ad.PTGENDER == 'Male']
    else:
        df = fe[fe['PTGENDER'] == 'Female']
        df2 = cn_mci[cn_mci.PTGENDER == 'Female']
        df3 = mci_ad[mci_ad.PTGENDER == 'Female']
        df4 = cn_ad[cn_ad.PTGENDER == 'Female']
        
    # use correct comparison depending on biomarker increase vs. decrease    
    if conf > 0:
        end_CN = len(df[(df['DX'] == 'CN') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])
        end_MCI = len(df[(df['DX'] == 'MCI') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])
        end_AD = len(df[(df['DX'] == 'AD') & (df[biomarker] > conf)]) / len(df[df[biomarker] > conf])     
        prog_CN_MCI = len(df2[df2[biomarker] > conf]) / len(df2)
        prog_MCI_AD = len(df3[df3[biomarker] > conf]) / len(df3)
        prog_CN_AD = len(df4[df4[biomarker] > conf]) / len(df4)
    else:
        end_CN = len(df[(df['DX'] == 'CN') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])
        end_MCI = len(df[(df['DX'] == 'MCI') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])
        end_AD = len(df[(df['DX'] == 'AD') & (df[biomarker] < conf)]) / len(df[df[biomarker] < conf])     
        prog_CN_MCI = len(df2[df2[biomarker] < conf]) / len(df2)
        prog_MCI_AD = len(df3[df3[biomarker] < conf]) / len(df3)
        prog_CN_AD = len(df4[df4[biomarker] < conf]) / len(df4)

    # print results
    print('Percent exceeding threshold that ended CN: ', round(end_CN*100,2), '%')
    print('Percent exceeding threshold that ended MCI: ', round(end_MCI*100,2), '%')
    print('Percent exceeding threshold that ended AD: ', round(end_AD*100,2), '%')
    print('Percent progressing CN to MCI exceeding threshold: ', round(prog_CN_MCI*100,2), '%')
    print('Percent Progressing MCI to AD exceeding threshold: ', round(prog_MCI_AD*100,2), '%')
    print('Percent Progressing CN to AD exceeding threshold: ', round(prog_CN_AD*100,2), '%')
#!/usr/bin/env python3
"""Data Wrangling Module for Alzheimer's Capstone 1.

This module accepts the ADNIMERGE.csv filename as input
and returns two dataframes clin_data and scan_data.

This module is not designed to run with other .csv files.
Suggested namespace is aw.
"""

def wrangle_adni(filename):
    """This function returns the two dataframes.

    Provide the ADNIMERGE.csv file as a string, and unpack
    the two dataframes clin_data and scan_data that are returned.
    """

    # ensure pandas availability for the function
    if 'pd' not in globals():
        import pandas as pd

    # read in the data to a pandas dataframe
    adni_full = pd.read_csv('ADNIMERGE.csv', dtype='object')

    # set the logical orders for the two diagnoses
    DX = ['CN', 'MCI', 'Dementia']
    DX_bl = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']

    # initialize empty dataframe
    adni = pd.DataFrame()

    # convert datatypes to categorical, datetime, and int
    adni['PTGENDER'] = pd.Categorical(adni_full.PTGENDER)
    adni['DX'] = pd.Categorical(adni_full.DX, ordered=True, categories=DX)
    adni['DX_bl'] = pd.Categorical(adni_full.DX_bl, ordered=True, categories=DX_bl)
    adni['EXAMDATE'] = pd.to_datetime(adni_full['EXAMDATE'])
    adni['EXAMDATE_bl'] = pd.to_datetime(adni_full['EXAMDATE_bl'])
    adni['PTEDUCAT'] = adni_full.PTEDUCAT.astype('int')
    adni['Month'] = adni_full.Month.astype('int')
    adni['RID'] = adni_full.RID.astype('int')

    # create a list of float data columns, loop and assign float dtypes
    floats = ['AGE', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'Hippocampus',
              'Ventricles', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG', 'AV45']

    # loop and assign dtypes
    for i in floats:
        adni[i] = adni_full[i].astype('float')

        # age has no baseline '_bl' equivalent
        if i == 'AGE':
            continue

        # every other column has a '_bl' equivalent to convert as well
        else:    
            y = i + '_bl'
            adni[y] = adni_full[y].astype('float')

    # drop columns with too much missing data
    adni.drop(labels=['FDG', 'FDG_bl', 'AV45', 'AV45_bl'], axis='columns', inplace=True)

    # add EXAMDATE to the index
    adni.set_index([adni.RID, adni.EXAMDATE], inplace=True)

    # sort the index
    adni.sort_index(inplace=True)

    # remove redundant columns
    adni.drop(['RID', 'EXAMDATE'], axis='columns', inplace=True)

    # drop rows with missing diagnoses
    adni_dx = adni.dropna(subset=['DX', 'DX_bl'])

    # isolate clinical data
    clinical = adni_dx.loc[:, 'EXAMDATE_bl':'RAVLT_immediate_bl']

    # remove rows missing any values
    clinical_rmv = clinical.dropna(how='any')

    # filter results to patients with multiple visits
    num_clin_exams = clinical_rmv.groupby('RID')['EXAMDATE_bl'].count()
    clin_filter = num_clin_exams[num_clin_exams > 1]
    clin_data = clinical_rmv.loc[clin_filter.index]

    # filter the scan data
    scans = adni_dx.loc[:, 'Hippocampus':]

    # add EXAMDATE_bl back in
    scans['EXAMDATE_bl'] = adni_dx.EXAMDATE_bl

    # filter for complete rows
    scans_rmv = scans.dropna(how='any')

    # filter results to patients with multiple visits
    num_scan_exams = scans_rmv.groupby('RID')['EXAMDATE_bl'].count()
    scan_filter = num_scan_exams[num_scan_exams > 1]
    scan_data = scans_rmv.loc[scan_filter.index]

    return clin_data, scan_data

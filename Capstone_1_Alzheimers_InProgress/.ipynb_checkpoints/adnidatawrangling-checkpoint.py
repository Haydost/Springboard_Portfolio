#!/usr/bin/env python3
"""Data Wrangling Module for Alzheimer's Capstone 1.

This module returns three dataframes adni_comp, clin_data, and scan_data.
No filename is needed as input. Uses/designed for 'ADNIMERGE.csv' only.

This module is not designed to run with other .csv files.
Suggested namespace is aw.
"""

def wrangle_adni():
    """This function returns the two dataframes.

    Unpack the two dataframes clin_data and scan_data that are returned.
    """

    # ensure pandas availability for the function
    if 'pd' not in globals():
        import pandas as pd

    # read in the data to a pandas dataframe
    adni_full = pd.read_csv('ADNIMERGE.csv', dtype='object')

    # set the logical orders for the two diagnoses
    DX = ['CN', 'MCI', 'AD']
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
    
    # calculate dynamic age
    adni_dx.loc[:, 'AGE_dynamic'] = adni_dx.AGE + (adni_dx.Month / 12)
    
    # create dataframe with only patients that have complete scan and clinical data
    adni_rmv = adni_dx.dropna(how='any')
    
    # filter those results to only patients with multiple visits
    num_comp_exams = adni_rmv.groupby('RID')['EXAMDATE_bl'].count()
    adni_comp_filter = num_comp_exams[num_comp_exams > 1]
    adni_comp = adni_rmv.loc[adni_comp_filter.index]

    # isolate clinical data
    clinical = pd.DataFrame()
    clinical = adni_dx.loc[:, ['EXAMDATE_bl', 'Month', 'PTGENDER', 'DX', 'DX_bl', 'PTEDUCAT', 'AGE', 'AGE_dynamic',
                              'CDRSB', 'CDRSB_bl', 'ADAS11', 'ADAS11_bl', 'ADAS13', 'ADAS13_bl', 'MMSE',
                              'MMSE_bl', 'RAVLT_immediate', 'RAVLT_immediate_bl']]
    
    # remove rows missing any values
    clinical_rmv = clinical.dropna(how='any')

    # filter results to patients with multiple visits
    num_clin_exams = clinical_rmv.groupby('RID')['EXAMDATE_bl'].count()
    clin_filter = num_clin_exams[num_clin_exams > 1]
    clin_data = clinical_rmv.loc[clin_filter.index]

    # filter the scan data
    scans = pd.DataFrame()
    scans = adni_dx.loc[:, ['EXAMDATE_bl', 'Month', 'PTGENDER', 'DX', 'DX_bl', 'PTEDUCAT', 'AGE', 'AGE_dynamic',
                           'Hippocampus', 'Hippocampus_bl', 'Ventricles', 'Ventricles_bl', 'WholeBrain', 'WholeBrain_bl',
                           'Entorhinal', 'Entorhinal_bl', 'MidTemp', 'MidTemp_bl', ]]

    # filter for complete rows
    scans_rmv = scans.dropna(how='any')

    # filter results to patients with multiple visits
    num_scan_exams = scans_rmv.groupby('RID')['EXAMDATE_bl'].count()
    scan_filter = num_scan_exams[num_scan_exams > 1]
    scan_data = scans_rmv.loc[scan_filter.index]

    return adni_comp, clin_data, scan_data
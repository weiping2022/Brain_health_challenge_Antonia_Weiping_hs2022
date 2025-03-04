import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt


#%% Defining variables / lists for merging files
medhist_files = ['MEDHIST', 'INITHEALTH', 'RECMHIST', 'RECBLLOG', 'BLSCHECK']
diag_files =  ['ADSXLIST', 'BLCHANGE']
key_attrs = ['Phase','RID','VISCODE2'] 
key_osa = ['apnea','sleep disordered breathing','SDB','OSA']   
files = [
    'ADSXLIST', 'BLCHANGE', 'DXSUM_PDXCONV_ADNIALL', 'GDSCALE', 'NEUROBAT', 'MEDHIST', 'INITHEALTH', 'RECMHIST', 'RECBLLOG', 'BLSCHECK'
]


# definition of possible values in VISCODE2
viscode2_type = CategoricalDtype(categories=['sc', 'scmri', 'bl', 'mc03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96', 'm108', 'm120', 'm132', 'm144', 'm156', 'm168', 'm180', 'f'], ordered=True)

#Definition of categories for the categorical columns
diag_type = CategoricalDtype(categories=['NL', 'MCI', 'AD'], ordered=True)
phase_type = CategoricalDtype(categories=['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], ordered=True)
gd_scores_type = CategoricalDtype(categories=['none', 'mild', 'moderate/severe'], ordered=True)
dx_severity_type = CategoricalDtype(categories=['mild', 'moderate', 'severe'], ordered=True)
dxconfid_type = CategoricalDtype(categories=['uncertain', 'mildly confident', 'moderately confident', 'highly confident'], ordered=True)

#definition of columns that only have binary values
binary_adsxlist_cols = ['AXABDOMN', 'AXSWEATN', 'AXDIZZY', 'AXENERGY',
    'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN',
    'AXDPMOOD']

binary_blchange_cols = ['BCADAS', 'BCMMSE', 'BCMMSREC', 'BCNMMMS', 'BCNEUPSY',
    'BCNONMEM', 'BCFAQ', 'BCCDR', 'BCDEPRES', 'BCSTROKE', 'BCDELIR',
    'BCEXTCIR', 'BCCORADL', 'BCCORCOG']


#%% compile mainfile function of CJK Team

def compile_mainfile(dest = './data/mainfile.csv'):
    '''
    Joins all tabular datasets into one huge dataset and saves it as CSV file
    at the desired location.

    Information about the datasets in general:
    https://adni.loni.usc.edu/wp-content/uploads/2008/07/inst_about_data.pdf

    Arguments
    ---------
    dest: File path where the final main dataset should be stored as CSV.

    Returns
    -------
    main_data: a pd.DataFrame containing the joined data
    '''

    # load PTDEMOG which builds the basis for our main dataset
    main_data = pd.read_csv('PTDEMOG.csv')

    # load file per file and join it with the already loaded data
    for file in files:
        additional_data = pd.read_csv(
            f'{file}.csv', low_memory=False)
        additional_data = additional_data.drop(
            columns=['ID', 'VISCODE', 'SITEID', 'USERDATE', 'USERDATE2', 'update_stamp'], errors='ignore')
        main_data = pd.merge(main_data, additional_data,
                             how='outer', on=key_attrs)

    main_data['Phase'] = main_data['Phase'].astype(phase_type)
    main_data['VISCODE2'] = main_data['VISCODE2'].astype(viscode2_type)
    main_data['USERDATE'] = pd.to_datetime(main_data['USERDATE'])
    main_data['USERDATE2'] = pd.to_datetime(main_data['USERDATE2'])

    main_data['PTSOURCE'] = main_data['PTSOURCE'].map({
        1: 'Participant Visit',
        2: 'Telephone Call'
    }).astype('category')
    main_data['PTGENDER'] = main_data['PTGENDER'].map({1: 'male', 2: 'female'}).astype('category')
    main_data['PTBIRTHMONTH'] = main_data[['PTDOBMM', 'PTDOBYY']].apply(lambda x: np.nan if x.isna().any() else pd.to_datetime(f"{x['PTDOBMM']:.0f}.{x['PTDOBYY']:.0f}", format = '%m.%Y'), axis = 1)
    main_data['PTHAND'] = main_data['PTHAND'].map({
        1: 'right',
        2: 'left',
    }).astype('category')
    main_data['PTMARRY'] = main_data['PTMARRY'].map({
        1: 'Married',
        2: 'Widowed',
        3: 'Divorced',
        4: 'Never married'
    }).astype('category')
    main_data.loc[main_data['PTEDUCAT'] < 0, ['PTEDUCAT']] = np.nan
    main_data['PTNOTRT'] = main_data['PTNOTRT'].replace(2, np.nan) # 2 == not applicable
    main_data['PTRTYR'] = pd.to_datetime(main_data['PTRTYR'].str.replace('--', '01'), format = '%m/%d/%Y', errors = 'coerce')
    main_data['PTHOME'] = main_data['PTHOME'].map({
        1: 'House',
        2: 'Condo/Co-op (owned)',
        3: 'Apartment (rented)',
        4: 'Mobile Home',
        5: 'Retirement Community',
        6: 'Assisted Living',
        7: 'Skilled Nursing Facility',
        8: 'Other'
    }).astype('category')
    main_data['PTTLANG'] = main_data['PTTLANG'].map({
        1: 'English',
        2: 'Spanish',
    }).astype('category')
    main_data['PTPLANG'] = main_data['PTPLANG'].map({
        1: 'English',
        2: 'Spanish',
        3: 'Other'
    }).astype('category')
    main_data['PTCOGBEG'] = main_data['PTCOGBEG'].replace(9999, np.nan)
    main_data['PTADBEG'] = main_data['PTADBEG'].replace(-1, np.nan)
    main_data['PTADDX'] = main_data['PTADDX'].replace(9999, np.nan)
    main_data['PTETHCAT'] = main_data['PTETHCAT'].replace(3, np.nan).replace(2, 0)
    main_data['PTRACCAT'] = main_data['PTRACCAT'].map({
        1: 'American Indian or Alaskan Native',
        2: 'Asian',
        3: 'Native Hawaiian or Other Pacific Islander',
        4: 'Black or African American',
        5: 'White',
        6: 'More than one race'
    }).astype('category')

    main_data['GDSOURCE'] = main_data['GDSOURCE'].map({
        1: 'Participant Visit',
        2: 'Telephone Call'
    }).astype('category')
    main_data.loc[main_data['GDTOTAL'] < 0, ['GDTOTAL']] = np.nan

    # combine the three columns with diagnosis to one new columns
    main_data.DXCHANGE = main_data.DXCHANGE.map({
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
        6: 3,
        7: 1,
        8: 2,
        9: 1,
    })
    main_data['DIAG'] = main_data.DXCURREN.fillna(main_data.DXCHANGE). fillna(main_data.DIAGNOSIS).map({1: 'NL', 2: 'MCI', 3: 'AD'})
    main_data = main_data.drop(['DXCHANGE', 'DXCURREN', 'DIAGNOSIS'], axis=1)
    main_data['DXCONV'] = main_data['DXCONV'].map({
        0: 'No',
        1: 'Conversion',
        2: 'Reversion'
    }).astype('category')
    main_data['DXCONTYP'] = main_data['DXCONTYP'].map({
        1: 'NC to MCI',
        2: 'NC to AD',
        3: 'MC to AD'
    }).astype('category')
    main_data['DXREV'] = main_data['DXREV'].map({
        1: 'MCI to NC',
        2: 'AD to MCI',
        3: 'AD to NC'
    }).astype('category')
    main_data['DXMDES'] = main_data['DXMDES'].map({
        1: 'Memory features',
        2: 'Non-memory features',
        '1:02': 'both',
        '1:2': 'both',
        '1|2': 'both'
    }).astype('category')
    main_data['DXMDUE'] = main_data['DXMDUE'].map({
        1: 'MCI due to Alzheimers Disease',
        2: 'MCI due to other etiology'
    }).astype('category')
    # these columns contain different values depending on the phase and
    # since we probably won't use them we can avoid the hussle to encode
    # them correctly by just simply dropping them
    main_data = main_data.drop(columns = ['DXMOTHET', 'DXODES'])
    main_data[['DXDSEV', 'DXADES']] = main_data[['DXDSEV', 'DXADES']].replace({
        1: 'mild',
        2: 'moderate',
        3: 'severe'
    }).astype(dx_severity_type)
    main_data['DXDDUE'] = main_data['DXDDUE'].map({
        1: 'Dementia due to Alzheimers Disease',
        2: 'Dementia due to other etiology'
    }).astype('category')
    main_data['DXAPP'] = main_data['DXAPP'].map({
        1: 'Probable',
        2: 'Possible'
    }).astype('category')
    main_data['DXAPROB'] = main_data['DXAPROB'].map({
        1: 'None',
        2: 'Strokes',
        3: 'Depression',
        4: 'Delirium',
        5: 'Parkinsonism',
        6: 'Metabolic/Toxic Disorder',
        7: 'Other'
    }).astype('category')
    main_data['DXAPOSS'] = main_data['DXAPOSS'].map({
        1: 'Atypical clinical course or features',
        2: 'Strokes',
        3: 'Depression',
        4: 'Delirium',
        5: 'Parkinsonism',
        6: 'Metabolic/Toxic Disorder',
        7: 'Other'
    }).astype('category')
    main_data['DXPDES'] = main_data['DXPDES'].map({
        1: 'Parkinsonism without cognitive impairment',
        2: 'Parkinsonism with cognitive impairment, not demented',
        3: 'Parkinsonism with cognitive impairment, demented',
        4: 'Atypical Parkinsonism'
    }).astype('category')
    main_data['DXPCOG'] = main_data['DXPCOG'].map({
        1: 'PD',
        2: 'PDD',
        3: 'DLB',
        4: 'PDAD'
    }).astype('category')
    main_data['DXPATYP'] = main_data['DXPATYP'].map({
        1: 'PSP',
        2: 'CBGD',
        3: 'OPCA',
        4: 'SND',
        5: 'Shy Drager',
        6: 'Vascular',
        7: 'Other'
    }).astype('category')
    main_data['DXCONFID'] = main_data['DXCONFID'].map({
        1: 'uncertain',
        2: 'mildly confident',
        3: 'moderately confident',
        4: 'highly confident'
    }).astype(dxconfid_type)

    main_data[binary_adsxlist_cols] = main_data[binary_adsxlist_cols].replace(1, 0).replace(2, 1).replace(-1, np.nan)

    main_data['BCPREDX'] = main_data['BCPREDX'].map({1: 'NL', 2: 'MCI', 3: 'AD'}).astype(diag_type)
    main_data[binary_blchange_cols] = main_data[binary_blchange_cols].replace(-1, np.nan)
    main_data.loc[:,'GDTOTAL_GROUP'] = pd.cut(main_data['GDTOTAL'], [-1,4,9,15], labels=list(gd_scores_type.categories))

    # mark missings
    main_data = main_data.replace(-4.0, np.nan)
    main_data = main_data.replace("-4", np.nan)
    # "-1" is also a missing value
    main_data = main_data.replace(-1, np.nan)
    # for some time data we have "9999" for missing values
    main_data = main_data.replace(9999, np.nan)

    # fill missing values first with the ones from previous visits and then
    # with upcoming visits.
    main_data = main_data.sort_values(['Phase', 'VISCODE2'])
    main_data.DIAG = main_data.groupby('RID').DIAG.ffill()
    main_data.DIAG = main_data.groupby('RID').DIAG.bfill()
    main_data = pd.concat([main_data[['RID', 'DIAG']], main_data.groupby(
        ['RID', 'DIAG'], as_index = False).ffill()], axis=1)
    main_data = pd.concat([main_data[['RID', 'DIAG']], main_data.groupby(
        ['RID', 'DIAG'], as_index = False).bfill()], axis=1)

    # add information about changes of diagnosis
    main_data['PREV_DIAG'] = main_data.groupby('RID', sort = False)['DIAG'].shift().fillna(main_data['DIAG'])
    main_data['DIAG_CHANGED'] = (~main_data['PREV_DIAG'].isna()) & (main_data['PREV_DIAG'] != main_data['DIAG'])
    main_data['DIAG_GROUP'] = (main_data.PREV_DIAG.astype(str) + '-' + main_data.DIAG.astype(str)).astype('category')
    main_data['DIAG_GROUP'] = main_data['DIAG_GROUP'].replace('nan-nan', np.nan).replace('None-None', np.nan)

    diag_data = main_data.groupby('RID').agg({'DIAG': ['first', 'last']}).reset_index()
    diag_data.columns = ['_'.join(col).strip() for col in diag_data.columns.values]
    diag_data = diag_data.rename(columns = {'RID_': 'RID', 'DIAG_first': 'PATIENT_FIRST_DIAG', 'DIAG_last': 'PATIENT_LAST_DIAG'})
    diag_data['PATIENT_DIAG_GROUP'] = (diag_data.PATIENT_FIRST_DIAG.astype(str) + '-' + diag_data.PATIENT_LAST_DIAG.astype(str)).astype('category')
    diag_data['PATIENT_DIAG_GROUP'] = diag_data['PATIENT_DIAG_GROUP'].replace('nan-nan', np.nan).replace('None-None', np.nan)
    diag_data['PATIENT_DIAG_CHANGED'] = diag_data['PATIENT_DIAG_GROUP'].isin(['NL-MCI', 'NL-AD', 'MCI-NL', 'MCI-AD', 'AD-NL', 'AD-MCI'])
    main_data = main_data.merge(diag_data, how = 'left', on = 'RID')

    main_data = main_data.sort_values(key_attrs)

    main_data.to_csv(dest, index = False)
    return main_data

compile_mainfile(dest = './data/mainfile.csv')
cached_main_data = None
#%% load mainfile function of CJK
def load_mainfile():
    '''
    Loads a stored version of the mainfile and returns it after setting the
    correct datatypes.

    Returns
    -------
    main_data: a pd.DataFrame containing the joined data
    '''

    global cached_main_data
    if cached_main_data is None:
        print('loading main data ...')

        main_data = pd.read_csv('./data/mainfile.csv', low_memory = False)
        main_data['Phase'] = main_data['Phase'].astype(phase_type)
        main_data['VISCODE2'] = main_data['VISCODE2'].astype(viscode2_type)
        main_data['USERDATE'] = pd.to_datetime(main_data['USERDATE'])
        main_data['USERDATE2'] = pd.to_datetime(main_data['USERDATE2'])
        main_data['DIAG'] = main_data['DIAG'].astype(diag_type)
        main_data['PREV_DIAG'] = main_data['DIAG'].astype(diag_type)
        main_data['DIAG_GROUP'] = main_data['DIAG_GROUP'].astype('category')
        main_data['PATIENT_FIRST_DIAG'] = main_data['PATIENT_FIRST_DIAG'].astype(diag_type)
        main_data['PATIENT_LAST_DIAG'] = main_data['PATIENT_LAST_DIAG'].astype(diag_type)
        main_data['PATIENT_DIAG_GROUP'] = main_data['PATIENT_DIAG_GROUP'].astype('category')

        main_data['PTSOURCE'] = main_data['PTSOURCE'].astype('category')
        main_data['PTGENDER'] = main_data['PTGENDER'].astype('category')
        main_data['PTBIRTHMONTH'] = pd.to_datetime(main_data['PTBIRTHMONTH'])
        main_data['PTHAND'] = main_data['PTHAND'].astype('category')
        main_data['PTMARRY'] = main_data['PTMARRY'].astype('category')
        main_data['PTRTYR'] = pd.to_datetime(main_data['PTRTYR'])
        main_data['PTHOME'] = main_data['PTHOME'].astype('category')
        main_data['PTTLANG'] = main_data['PTTLANG'].astype('category')
        main_data['PTPLANG'] = main_data['PTPLANG'].astype('category')
        main_data['PTRACCAT'] = main_data['PTRACCAT'].astype('category')

        #main_data[cdr_scores_cols] = main_data[cdr_scores_cols].astype(cdr_scores_type)
        main_data['CDSOURCE'] = main_data['CDSOURCE'].astype('category')
        main_data['CDVERSION'] = main_data['CDVERSION'].astype('category')

        main_data['MMRECALL'] = main_data['MMRECALL'].astype('category')
        main_data['WORDLIST'] = main_data['WORDLIST'].astype('category')
#        main_data['MMSCORE_GROUP'] = main_data['MMSCORE_GROUP'].astype(mmse_scores_type)

      #  main_data[delw_cols] = main_data[delw_cols].astype('category')

        main_data['LMSTORY'] = main_data['LMSTORY'].astype('category')

        main_data['GDSOURCE'] = main_data['GDSOURCE'].astype('category')
        main_data['GDTOTAL_GROUP'] = main_data['GDTOTAL_GROUP'].astype(gd_scores_type)

        main_data['DXCONV'] = main_data['DXCONV'].astype('category')
        main_data['DXCONTYP'] = main_data['DXCONTYP'].astype('category')
        main_data['DXREV'] = main_data['DXREV'].astype('category')
        main_data['DXMDES'] = main_data['DXMDES'].astype('category')
        main_data['DXMDUE'] = main_data['DXMDUE'].astype('category')
        main_data[['DXDSEV', 'DXADES']] = main_data[['DXDSEV', 'DXADES']].astype(dx_severity_type)
        main_data['DXDDUE'] = main_data['DXDDUE'].astype('category')
        main_data['DXAPP'] = main_data['DXAPP'].astype('category')
        main_data['DXAPROB'] = main_data['DXAPROB'].astype('category')
        main_data['DXAPOSS'] = main_data['DXAPOSS'].astype('category')
        main_data['DXPDES'] = main_data['DXPDES'].astype('category')
        main_data['DXPCOG'] = main_data['DXPCOG'].astype('category')
        main_data['DXPATYP'] = main_data['DXPATYP'].astype('category')
        main_data['DXCONFID'] = main_data['DXCONFID'].astype(dxconfid_type)

        main_data['BCPREDX'] = main_data['BCPREDX'].astype(diag_type)

        main_data['update_stamp'] = pd.to_datetime(main_data['update_stamp'])

        cached_main_data = main_data

    return cached_main_data.copy()


if __name__ == "__main__":
    compile_mainfile()

compile_mainfile()


#%% various functions WZ

# check if one of the keyword is in a string
def key_in_str(key_list,str):
    if any(word in str for word in key_list):
        new_val = 1
    else: new_val = 0
    return new_val

# generate a new column with value: 1 if a row includes one of the keywords, 0 if not includes
def new_col_with_key(df,key_list,new_col):
    #df = pd.read_csv(file,sep=';')
    df = df.astype('string')
    df['new'] = df[df.columns].apply(lambda x: '_'.join(x.dropna()), axis=1)
    L = []
    for str in df['new']:
        L.append(key_in_str(key_list,str))
    df[new_col] = L
    return df[key_attrs + [new_col]]

def drop_char(df,col): # drop the row where strings in a column cant be converted to number, and convert the rest to integer
    dff = df.copy()
    col_lst = dff[col].tolist() # extract the column to be a list
    index_lst = []  # index_l to store the index which should be droped
    for i in range(len(col_lst)):  
        if pd.notna(col_lst[i]): 
            str_ = col_lst[i]
            try:
                int(str_)
            except (RuntimeError, TypeError, NameError,ValueError):
                index_lst.append(i)  
    dff = dff.drop(index_lst)
    
    lst = dff[col].to_list()
    int_lst = []
    for i in range(len(lst)):
        to_int = int(lst[i])
        int_lst.append(to_int)
    dff[col] = int_lst
    return dff.reset_index()

def char_float_na(df,col): # replace the strings in a column which cant be converted to float number, and convert the rest to float
    dff = df.copy()
    col_lst = dff[col].tolist() # extract the column to be a list
    index_lst = []  # index_l to store the index which should be droped
    for i in range(len(col_lst)):  
        if pd.notna(col_lst[i]): 
            str_ = col_lst[i]
            try:
                float(str_)
            except (RuntimeError, TypeError, NameError,ValueError):
                dff=dff.replace({str_:np.NaN})
    return dff

#%%

df = pd.read_csv('./data/mainfile.csv')

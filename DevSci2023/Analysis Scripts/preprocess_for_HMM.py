#!/usr/bin/env python3

"""
Created on Wed May 11 10:23:45 2022

@author: Susan Benear slb671@nyu.edu

Code adapted from Samantha Cohen
https://github.com/samsydco/HBN/blob/master/4_Preprocess_HPC.py

"""

import glob
import nibabel as nib
import pandas as pd
import numpy as np
import os
import h5py
from sklearn import linear_model
from scipy import stats
from nilearn.masking import apply_mask

# set up directories
bids_dir = '/data/projects/learning_lemurs/'
fmriprep_dir = os.path.join(bids_dir, 'derivatives/fmriprep/')

#define subjects and ROIs
subs = ['sub-1000','sub-1001','sub-1002','sub-1006','sub-1007','sub-1009','sub-1011','sub-1012','sub-1015','sub-1016','sub-1017','sub-1019','sub-1023','sub-1024','sub-1025','sub-1027','sub-1028','sub-1029','sub-1031','sub-1032','sub-1033','sub-1034','sub-1035','sub-1036','sub-1037','sub-1038','sub-1039','sub-1040','sub-1045','sub-1048','sub-1050','sub-1052','sub-1053','sub-1054','sub-1056','sub-1058','sub-3000']
#subs = ['sub-1000']
rois = ['PCC','LAG','RAG','LHC','RHC']
#rois=['vmPFC']

for sub in subs:
    print('Processing subject ', sub)
    # label fmri task(s) for creating groups in the output file later
    task = 'AHKJ'
    # path to the functional data (smoothed, but not denoised)
    func_data = os.path.join(bids_dir, 'fsloutput', 'preprocessed_'+sub+'.feat','filtered_func_data.nii.gz')
    
    for roi in rois:    
        # path to the masks for each roi
        roi_mask = os.path.join(bids_dir + 'rsa/'+roi+'_thrbin_ped.nii.gz')
        #apply the mask to the functional data
        roi_data = apply_mask(func_data, roi_mask)
        print('masked '+roi+' data shape is ', roi_data.shape)

   
        print('Adding in confounds')
        # path to the subject's confounds file (output from fmriprep)
        conf = np.genfromtxt(os.path.join(fmriprep_dir + sub + '/func/' + sub + '_task-AHKJep2_run-1_desc-confounds_timeseries.tsv'), names=True)
        # create a motion variable containing the 6 motion parameters
        motion = np.column_stack((conf['trans_x'],
                            conf['trans_y'],
                            conf['trans_z'],
                            conf['rot_x'],
                            conf['rot_y'],
                            conf['rot_z']))
        # create a variable containing other noise paramters (WM & CSF, cosine parameters aka high pass filters)
        # and stack these next to the motion parameters
        reg = np.column_stack((conf['csf'],
                        conf['white_matter'],
                        np.nan_to_num(conf['framewise_displacement']),
                        np.column_stack([conf[k] for k in conf.dtype.names if 'cosine' in k]),
                        motion,
                        np.vstack((np.zeros((1,motion.shape[1])), np.diff(motion, axis=0)))))
                           
        print('      Cleaning and zscoring')
        # set up the linear regression
        regr = linear_model.LinearRegression()
        # fit the regression to the masked data
        regr.fit(reg, roi_data)
        # set the data equal to the transposed version of itself minus the regressed out parameters
        roi_data = roi_data.T - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
        # z score the data
        roi_data = stats.zscore(roi_data, axis=1)
        # transpose the data one last time so it ends up being TRs x voxels, which is what HMM wants as input
        roi_data = roi_data.T
        # set the data equal to the name of the roi currently being run so that it doesn't overwrite the previous roi
        # (there is definitely a more elegant way to do this, but this is what I came up with)
        if roi == 'PCC':
            PCC_data = roi_data
        elif roi == 'vmPFC':
            vmPFC_data = roi_data
        elif roi == 'LAG':
            LAG_data = roi_data
        elif roi == 'RAG':
            RAG_data = roi_data 
        elif roi == 'LHC':
            LHC_data = roi_data
        elif roi == 'RHC':
            RHC_data = roi_data

    # create an .h5 file for each participant to store all the output for each ROI hierarchically 
    # under each task (in this case just one task)
    #IF YOU WANT TO CREATE THE h5 FILES:
    with h5py.File(os.path.join(bids_dir,'hmm/h5files/' + sub + '.h5')) as hf:
        grp = hf.create_group(task)
        grp.create_dataset('PCC', data=PCC_data)
       #grp.create_dataset('vmPFC', data=vmPFC_data)
        grp.create_dataset('LAG', data=LAG_data)
        grp.create_dataset('RAG', data=RAG_data)
        grp.create_dataset('LHC', data=LHC_data)
        grp.create_dataset('RHC', data=RHC_data)
        grp.create_dataset('reg',data=reg)
        print('done creating h file for ', sub)
        #print(hf.shape)
        #print(hf)
#END
        
    #IF YOU MESS UP AND NEED TO DELETE AND REPLACE THE DATA IN AN h5 FILE, run this instead of the "with h5py" code above:
    with h5py.File(os.path.join(bids_dir,'hmm/h5files/' + sub + '.h5'), 'a') as hf:
        del hf['AHKJ']['RHC']
        hf.create_dataset('/AHKJ/RHC', data=RHC_data)
        print('done with h file for ' +sub)  

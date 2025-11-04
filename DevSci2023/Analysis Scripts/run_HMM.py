#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:23:45 2022

@author: Susan Benear slb671@nyu.edu

Code adapted from Chris Baldassano and Linda Geerligs
https://naturalistic-data.org/content/Event_Segmentation.html

"""
#load in packages
import warnings
import sys 
import os    
import glob
from statesegmentation import GSBS
from functools import reduce
import numpy as np
from brainiak.eventseg.event import EventSegment
import nibabel as nib
from nilearn.masking import apply_mask
from nltools.data import Brain_Data
from scipy.stats import zscore, norm
import datalad.api as dl
import deepdish as dd

from matplotlib import pyplot as plt
import matplotlib.patches as patches

#defining parameters for figures later
smallsize=14; mediumsize=16; largesize=18
plt.rc('xtick', labelsize=smallsize); plt.rc('ytick', labelsize=smallsize); plt.rc('legend', fontsize=mediumsize)
plt.rc('figure', titlesize=largesize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)

###set up subs, rois, and data directories###

subs_list = ['sub-1000','sub-1001','sub-1002','sub-1006','sub-1007','sub-1009','sub-1011','sub-1012','sub-1015','sub-1016','sub-1017','sub-1019','sub-1023','sub-1024','sub-1025','sub-1027','sub-1028','sub-1029','sub-1031','sub-1032','sub-1033','sub-1034','sub-1035','sub-1036','sub-1038','sub-1039','sub-1040','sub-1045','sub-1048','sub-1050','sub-1052','sub-1053','sub-1054','sub-1056','sub-1058','sub-3000'] #create your list of subjects
rois = ['PCC','V1','LHC','RHC','LAG','RAG']

data_dir = '/data/projects/learning_lemurs/fsloutput/'
project_dir = '/data/projects/learning_lemurs/'

#create empty dictionaries to populate later
roi_dict = {}
movie_dict = {}
for roi in rois:
    #create empty list to populate later
    movie = []
    for sub in subs_list:
        #load in participants' h5 files we create using "hmm_preprocess.py"
        h5data = dd.io.load(os.path.join(project_dir, 'hmm/h5files/'+sub+'.h5'))
        #load in the participants data for this roi
        masked_data = h5data['AHKJ'][roi]
        #I had some issues with empty data in some ROIs, so used this to identify which ROIs for which pp had NaNs
        if np.isnan(masked_data).any() == True:
            print(roi+'for '+sub+'contains NaNs')
        #add this pp data to the movie list for this roi
        movie.append(masked_data)
    #pulling only the valid voxels? Tbh not clear on what's happening here
    valid_vox = reduce(np.union1d, [np.where(np.std(m, axis=0)>0)[0] for m in movie])
    movie = [m[:,valid_vox] for m in movie]
    #save out a numpy file to be used later?
    np.save(os.path.join(project_dir, 'hmm','AHKJ_'+roi+'_viewingWB.npy'), movie)

    print('Creating moviegroup variable for ',roi)
    movie = np.load(os.path.join(project_dir, 'hmm', 'AHKJ_'+roi+'_viewingWB.npy'))
    movie_group = np.mean(movie, axis=0) 
    
    if roi == 'PCC':
        roi_dict['PCC'] = movie_group
        movie_dict['PCC'] = movie
    elif roi == 'vmPFC':
        roi_dict['vmPFC'] = movie_group
        movie_dict['vmPFC'] = movie
    elif roi == 'V1':
        roi_dict['V1'] = movie_group  
        movie_dict['V1'] = movie
    elif roi == 'LAG':
        roi_dict['LAG'] = movie_group
        movie_dict['LAG'] = movie
    elif roi == 'RAG':
        roi_dict['RAG'] = movie_group 
        movie_dict['RAG'] = movie
    elif roi == 'LHC':
        roi_dict['LHC'] = movie_group
        movie_dict['LHC'] = movie
    elif roi == 'RHC':
        roi_dict['RHC'] = movie_group    
        movie_dict['RHC'] = movie


###plot the correlation between activity patterns for each pair of timepoints during the movie###
#groups = [PCC_group, vmPFC_group, V1_group, LAG_group, RAG_group, LHC_group, RHC_group]

for roi in roi_dict.keys():
    plt.figure(figsize=(10,8))
    plt.imshow(np.corrcoef(roi_dict[roi]))
    plt.xlabel('Timepoint')
    plt.ylabel('Timepoint')
    plt.colorbar()
    plt.title('Spatial pattern correlation for '+ roi);
    
    
    
    
    
#look at the  brain activity in each ROI based on a bunch of different metrics    
bounds_dict = {}
for roi in roi_dict.keys():
    
    #use an HMM to find both the event timings and the patterns corresponding to each event
    #specify number of events as mean number BPs behaviorally
    movie_HMM = EventSegment(n_events = 11)
    movie_HMM.fit(roi_dict[roi]);

    # Plotting the log-likelihood (measuring overall model fit)
   # plt.figure(figsize = (12, 4))
   # plt.plot(movie_HMM.ll_)
   # plt.title('Log likelihood during training for '+ roi)
   # plt.xlabel('Model fitting steps')

    # Plotting mean activity in each event for some example voxels
   # plt.figure(figsize = (12, 4))
   # example_vox = np.arange(0,movie_HMM.event_pat_.shape[0],100)
   # plt.imshow(movie_HMM.event_pat_[example_vox,:], aspect='auto')
    #plt.xlabel('Event number')
    #plt.ylabel('Example Voxels for '+ roi)


    # Plot probability of being in each event at each timepoint
 #   plt.figure(figsize = (12, 6))
  #  plt.matshow(movie_HMM.segments_[0].T, aspect='auto')
   # plt.gca().xaxis.tick_bottom()
   # plt.colorbar()
    #plt.title('Event probability for '+ roi)

    # Identify event boundaries as timepoints when max probability switches events
    event_bounds = np.where(np.diff(np.argmax(movie_HMM.segments_[0], axis = 1)))[0]
    nTRs = roi_dict[roi].shape[0]
    
    if roi == 'PCC':
        bounds_dict['PCC'] = event_bounds
    elif roi == 'vmPFC':
        bounds_dict['vmPFC'] = event_bounds
    elif roi == 'V1':
        bounds_dict['V1'] = event_bounds
    elif roi == 'LAG':
        bounds_dict['LAG'] = event_bounds
    elif roi == 'RAG':
        bounds_dict['RAG'] = event_bounds
    elif roi == 'LHC':
        bounds_dict['LHC'] = event_bounds
    elif roi == 'RHC':
        bounds_dict['RHC'] = event_bounds

for roi in roi_dict.keys():
    for bounds in bounds_dict.keys():
        if roi == bounds:
    # Plot boundaries as boxes on top of timepoint correlation matrix
            def plot_tt_similarity_matrix(ax, data_matrix, bounds, n_TRs, title_text):
    
                ax.imshow(np.corrcoef(data_matrix), cmap = 'viridis')
                ax.set_title(title_text)
                ax.set_xlabel('TR')
                ax.set_ylabel('TR')
    
        # plot the boundaries 
                bounds_aug = np.concatenate(([0], bounds, [n_TRs]))
    
                for i in range(len(bounds_aug) - 1):
                    rect = patches.Rectangle(
                        (bounds_aug[i], bounds_aug[i]),
                        bounds_aug[i+1] - bounds_aug[i],
                        bounds_aug[i+1] - bounds_aug[i],
                        linewidth = 2, edgecolor = 'w',facecolor = 'none'
                        )
                    ax.add_patch(rect)


            f, ax = plt.subplots(1,1, figsize = (10,8))
            title_text = 'Overlay the HMM-predicted event boundaries for '+roi+'\n on top of the TR-TR correlation matrix'
            plot_tt_similarity_matrix(ax, roi_dict[roi], bounds_dict[bounds], nTRs, title_text)



######################################################################################################
#Better fit for more heterogenous event lengths

#to determine the best number of events, fit the model on a training set 
#and then test the model fit on independent subjects
#I changed # subs to 18 from 8 (roughly half my number of participants)

test_dict = {}
for movies in movie_dict.keys():
    
    movie = movie_dict[movies]

    k_array = np.arange(20, 101, 10)
    test_ll = np.zeros(len(k_array))

    for i, k in enumerate(k_array):
        print('Trying %d events' % k)
    
        print('   Fitting model on training subjects for '+str(movies)+'...')
        movie_train = np.mean(movie[:18], axis = 0)
        movie_HMM = EventSegment(k)
        movie_HMM.fit(movie_train)
    
        print('   Testing model fit on held-out subjects for '+str(movies) +'...')
        movie_test = np.mean(movie[18:], axis = 0)
        _, test_ll[i] = movie_HMM.find_events(movie_test)
        
        print('inner loop done')

        if movies == 'PCC':
            test_dict['PCC'] = test_ll
        elif movies == 'V1':
            test_dict['V1'] = test_ll
        elif movies == 'vmPFC':
            test_dict['vmPFC'] = test_ll
        elif movies == 'RAG':
            test_dict['RAG'] = test_ll
        elif movies == 'LAG':
            test_dict['LAG'] = test_ll
        elif movies == 'RHC':
            test_dict['RHC'] = test_ll
        elif movies == 'LHC':
            test_dict['LHC'] = test_ll
            
    print('middle loop done')
    
print('outside loop done')


    

plt.plot(k_array, test_dict['RAG'])
plt.xlabel('Number of events for RAG')
plt.ylabel('Log-likelihood')

movie_dur = nTRs * 0.8  # Data acquired every 0.8 seconds
secax = plt.gca().secondary_xaxis('top', functions=(lambda x: movie_dur / x, lambda x: movie_dur / x))
secax.set_xlabel('Average event length (sec)')#+str(test))



#more flexibility in the HMM
#In addition to the “vanilla” HMM, we’ll run an HMM with more flexibility 
#during fitting (allowing for split-merge operations). This is slower 
#but can produce better fits if events are very uneven in duration. 
#We will use these segmentations below for comparison with an alternative 
#event segmentation method (GSBS) and with human labeled event boundaries.

#here I'm fitting 21 events to yield 20 boundaries to compare to my 20 behavioral boundaries, but
#this can be literally any number you want, including whatever the loglikehood function above identified
#as the optimal number of events for each ROI, which is what you could do with the conditional "if"
#statements here for each ROI (currently commented out)
hmm_dict = {}
hmm_sm_dict = {}
for roi in roi_dict.keys():
    
   # if roi == 'PCC' or roi == 'V1':

        print('Fitting HMM with 21 events...')
        HMM40 = EventSegment(n_events = 21)
        HMM40.fit(roi_dict[roi])
        HMM40_bounds = np.where(np.diff(np.argmax(HMM40.segments_[0], axis = 1)))[0]

        print('Fitting split-merge HMM with 21 events...')
        HMM40_SM = EventSegment(n_events = 21, split_merge = True)
        HMM40_SM.fit(roi_dict[roi])
        HMM40_SM_bounds = np.where(np.diff(np.argmax(HMM40_SM.segments_[0], axis = 1)))[0]     
    
        if roi == 'PCC':
            hmm_dict['PCC'] = HMM40_bounds
            hmm_sm_dict['PCC'] = HMM40_SM_bounds
        elif roi == 'V1':
           hmm_dict['V1'] = HMM40_bounds
           hmm_sm_dict['V1'] = HMM40_SM_bounds    
        elif roi == 'RAG':
           hmm_dict['RAG'] = HMM40_bounds
           hmm_sm_dict['RAG'] = HMM40_SM_bounds
        elif roi == 'LAG':
           hmm_dict['LAG'] = HMM40_bounds
           hmm_sm_dict['LAG'] = HMM40_SM_bounds
        if roi == 'vmPFC':
            hmm_dict['vmPFC'] = HMM40_bounds
            hmm_sm_dict['vmPFC'] = HMM40_SM_bounds
        elif roi == 'RHC':
            hmm_dict['RHC'] = HMM40_bounds
            hmm_sm_dict['RHC'] = HMM40_SM_bounds
        elif roi == 'LHC':
            hmm_dict['LHC'] = HMM40_bounds
            hmm_sm_dict['LHC'] = HMM40_SM_bounds
            
#saved these HMM-identified boundaries out to csv's            
import pandas as pd
df = pd.DataFrame({k:pd.Series(v) for k,v in hmm_dict.items()})
df1 = pd.DataFrame({k:pd.Series(v) for k,v in hmm_sm_dict.items()})
df.to_csv('/data/projects/learning_lemurs/hmm/hmm_dict_20.csv')
df1.to_csv('/data/projects/learning_lemurs/hmm/hmm_sm_dict_20.csv')

#here's code you'd include for other ROIs for which you want to identify a different # of events            
    elif roi == 'LAG' or roi == 'RAG':

        print('Fitting HMM with 60 events...')
        HMM40 = EventSegment(n_events = 60)
        HMM40.fit(roi_dict[roi])
        HMM40_bounds = np.where(np.diff(np.argmax(HMM40.segments_[0], axis = 1)))[0]

        print('Fitting split-merge HMM with 60 events...')
        HMM40_SM = EventSegment(n_events = 60, split_merge = True)
        HMM40_SM.fit(roi_dict[roi])
        HMM40_SM_bounds = np.where(np.diff(np.argmax(HMM40_SM.segments_[0], axis = 1)))[0]     
    
        if roi == 'RAG':
            hmm_dict['RAG'] = HMM40_bounds
            hmm_sm_dict['RAG'] = HMM40_SM_bounds
        elif roi == 'LAG':
            hmm_dict['LAG'] = HMM40_bounds
            hmm_sm_dict['LAG'] = HMM40_SM_bounds

    elif roi == 'vmPFC' or roi == 'RHC' or roi == 'LHC':

        print('Fitting HMM with 30 events...')
        HMM40 = EventSegment(n_events = 30)
        HMM40.fit(roi_dict[roi])
        HMM40_bounds = np.where(np.diff(np.argmax(HMM40.segments_[0], axis = 1)))[0]

        print('Fitting split-merge HMM with 30 events...')
        HMM40_SM = EventSegment(n_events = 30, split_merge = True)
        HMM40_SM.fit(roi_dict[roi])
        HMM40_SM_bounds = np.where(np.diff(np.argmax(HMM40_SM.segments_[0], axis = 1)))[0]     
    
 
        if roi == 'vmPFC':
            hmm_dict['vmPFC'] = HMM40_bounds
            hmm_sm_dict['vmPFC'] = HMM40_SM_bounds
        elif roi == 'RHC':
            hmm_dict['RHC'] = HMM40_bounds
            hmm_sm_dict['RHC'] = HMM40_SM_bounds
        elif roi == 'LHC':
            hmm_dict['LHC'] = HMM40_bounds
            hmm_sm_dict['LHC'] = HMM40_SM_bounds






###comparing behavioral to neural boundaries####

#old boundaries:
#adult_bounds20 = [84,158,186,241,369,395,463,519,621,660,750,766,831,991,1084,1110,1209,1346,1520,1556]
#kid_bounds20 = [64,86,184,239,266,371,400,520,546,604,750,849,875,916,971,980,995,1124,1140,1285]


#new behavioral boundaries:
adult_bounds20 = [84, 159, 186, 241, 370, 396, 464, 519, 623, 661, 749, 766, 844, 1018, 1093, 1110, 1209, 1235, 1384, 1558]
kid_bounds20 = [64, 86, 184, 239, 266, 371, 400, 519, 610, 629, 848, 876, 953, 988, 1020, 1143, 1209, 1386, 1556, 1571]
#make these lists into numpy arrays
adult_bounds20 = np.array(adult_bounds20)
kid_bounds20 = np.array(kid_bounds20)


#COMPARE TO ADULT BOUNDS
compare_dict = {}
print('doing 1,000 HMM permutations')
for region in hmm_dict.keys():
# Computes fraction of "ground truth" bounds are covered by a set of proposed bounds
# Returns z score relative to a null distribution via permutation
    hmm = hmm_dict[region]
    hmm_sm = hmm_sm_dict[region]
    
    def match_z(proposed_bounds, gt_bounds, num_TRs):
        nPerm = 1000
        threshold = 3
        np.random.seed(0)

        gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
        match = np.zeros(nPerm + 1)
        for p in range(nPerm + 1):
            gt_bounds = np.cumsum(gt_lengths)[:-1]
            for b in gt_bounds:
                if np.any(np.abs(proposed_bounds - b) <= threshold):
                    match[p] += 1
            match[p] /= len(gt_bounds)
            gt_lengths = np.random.permutation(gt_lengths)
    
        return (match[0]-np.mean(match[1:]))/np.std(match[1:])
        print(match)

    bound_types = [hmm, hmm_sm, adult_bounds20]
    matchz_mat = np.zeros((len(bound_types), len(bound_types)))
    for i, b1 in enumerate(bound_types):
        for j, b2 in enumerate(bound_types):
            if i != j:
                matchz_mat[i,j] = match_z(b1, b2, nTRs)

    if region == 'PCC':
        compare_dict['PCC'] = matchz_mat
    elif region == 'V1':
        compare_dict['V1'] = matchz_mat
    #elif region == 'vmPFC':
        #compare_dict['vmPFC'] = matchz_mat
    elif region == 'RAG':
        compare_dict['RAG'] = matchz_mat
    elif region == 'LAG':
        compare_dict['LAG'] = matchz_mat
    elif region == 'RHC':
        compare_dict['RHC'] = matchz_mat
    elif region == 'LHC':
        compare_dict['LHC'] = matchz_mat

#makes a plot to show the significance of each comparison
for region in compare_dict.keys():
    mm = compare_dict[region]
    mm[np.diag(np.ones(3, dtype=bool))] = 0
    f, ax = plt.subplots(1,1, figsize = (3,5.7))
    a=plt.imshow(mm)
    plt.colorbar()
    plt.xticks(np.arange(0.5,3.5,1), [region+'_HMM', region+'_HMM_SM','Adult20'],rotation=45)
    plt.yticks(np.arange(0.5,3.5,1), [region+'_HMM', region+'_HMM_SM','Adult20'])
    plt.xlabel('Ground truth bounds')
    plt.ylabel('Proposed bounds')
    plt.title('Match vs. null (z)')
    a.set_extent([0,3,3,0])
#prints the p value for each behavioral to HMM comparison
    print('P values for '+region+' matches to Adult bounds: ')
    print('HMM=%f, HMM_SM=%f' % tuple(norm.sf(mm[:2,2]).tolist()))
    

#COMPARE TO KID BOUNDS
compare_dict = {}
print('doing 1,000 HMM permutations')
for region in hmm_dict.keys():
# Computes fraction of "ground truth" bounds are covered by a set of proposed bounds
# Returns z score relative to a null distribution via permutation
    hmm = hmm_dict[region]
    hmm_sm = hmm_sm_dict[region]
    
    def match_z(proposed_bounds, gt_bounds, num_TRs):
        nPerm = 1000
        threshold = 3
        np.random.seed(0)

        gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
        match = np.zeros(nPerm + 1)
        for p in range(nPerm + 1):
            gt_bounds = np.cumsum(gt_lengths)[:-1]
            for b in gt_bounds:
                if np.any(np.abs(proposed_bounds - b) <= threshold):
                    match[p] += 1
            match[p] /= len(gt_bounds)
            gt_lengths = np.random.permutation(gt_lengths)
    
        return (match[0]-np.mean(match[1:]))/np.std(match[1:])
        #print(match)


    bound_types = [hmm, hmm_sm, kid_bounds20]
    matchz_mat = np.zeros((len(bound_types), len(bound_types)))
    for i, b1 in enumerate(bound_types):
        for j, b2 in enumerate(bound_types):
            if i != j:
                matchz_mat[i,j] = match_z(b1, b2, nTRs)

    if region == 'PCC':
        compare_dict['PCC'] = matchz_mat
    elif region == 'V1':
        compare_dict['V1'] = matchz_mat
    #elif region == 'vmPFC':
        #compare_dict['vmPFC'] = matchz_mat
    elif region == 'RAG':
        compare_dict['RAG'] = matchz_mat
    elif region == 'LAG':
        compare_dict['LAG'] = matchz_mat
    elif region == 'RHC':
        compare_dict['RHC'] = matchz_mat
    elif region == 'LHC':
        compare_dict['LHC'] = matchz_mat

#makes a plot to show the significance of each comparison
for region in compare_dict.keys():
    mm = compare_dict[region]
    mm[np.diag(np.ones(3, dtype=bool))] = 0
    f, ax = plt.subplots(1,1, figsize = (3,5.7))
    a=plt.imshow(mm)
    plt.colorbar()
    plt.xticks(np.arange(0.5,3.5,1), [region+'_HMM', region+'_HMM_SM','Kid20'],rotation=45)
    plt.yticks(np.arange(0.5,3.5,1), [region+'_HMM', region+'_HMM_SM','Kid20'])
    plt.xlabel('Ground truth bounds')
    plt.ylabel('Proposed bounds')
    plt.title('Match vs. null (z)')
    a.set_extent([0,3,3,0])
#prints the p value for each behavioral to HMM comparison
    print('P values for '+region+' matches to Kid bounds: ')
    print('HMM=%f, HMM_SM=%f' % tuple(norm.sf(mm[:2,2]).tolist()))
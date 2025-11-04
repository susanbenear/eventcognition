#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created and modified Spring 2022

@author: Susan Benear slb671@nyu.edu

Code adapted from Haroon Popal via Mark Thorton
https://dartbrains.org/content/RSA.html
 
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from nltools.data import Brain_Data, Adjacency
from nltools.mask import expand_mask
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation
from sklearn.metrics import pairwise_distances
from nilearn.plotting import plot_glass_brain, plot_stat_map
from scipy.stats import spearmanr, ttest_1samp

#***FOR KIDS, CHANGE OUTPUT DIR, FEAT DIRECTORY, CSV LABEL, AND FILE LOCATION FOR CSVs TO SAVE***

# SET UP DIRECTORIES

bids_dir = '/data/projects/learning_lemurs/'
output_dir = os.path.join(bids_dir, 'rsa','40EVs_kids/')
os.chdir(bids_dir) #change directories to this one

#SUB 1037 REMOVED BC OF DROUPT IN LHC
subjs_list = ['sub-1000','sub-1001','sub-1002','sub-1006','sub-1007','sub-1009','sub-1011','sub-1012','sub-1015','sub-1016','sub-1017','sub-1019','sub-1023','sub-1024','sub-1025','sub-1027','sub-1028','sub-1029','sub-1031','sub-1032','sub-1033','sub-1034','sub-1035','sub-1036','sub-1038','sub-1039','sub-1040','sub-1045','sub-1048','sub-1050','sub-1052','sub-1053','sub-1054','sub-1056','sub-1058','sub-3000'] #create your list of subjects
rois = ['PCC','V1','LHC','RHC','LAG','RAG']

subjs_list.sort() #sort the list into numerical order, in order by occurence across the timeseries
print('Found '+str(len(subjs_list))+' subjects') #print out how many subjects were found

# SET UP LABELS FOR VISUALIZATIONS
conditions = pd.read_csv(bids_dir+'rsa/conditions_40.csv') #import a csv containing the list of conditions (18 within & 18 across time windows)
labels = conditions.label.tolist() #pull the label column and create a list containing the values from that column of the csv
#bring in these labels for type of pair and whether the comparison is within or boundary, to be used later
pairtypeconditions = pd.read_csv(bids_dir+'rsa/pairtype_kids.csv')
pairtypelabels = pairtypeconditions.pairtype.tolist()
withbounconditions = pd.read_csv(bids_dir+'rsa/withboun.csv')
withbounlabels = withbounconditions.withboun.tolist()

# HYPOTHESIS RDM
rdmhyp = pd.DataFrame(index=labels, columns=labels) #setting up a hypothesis matrix with the x and y axes both being the list of 26 labels
rdmhyp = rdmhyp.fillna(1) #fill the matrix with 1s
filter_col = [col for col in rdmhyp if col.startswith('With')]
#replace the 1s with 0s in all the cells whose columns start with "With"
for col in filter_col:
  for row in filter_col:
    if col[:6] == row[:6]:
      rdmhyp.loc[row,col] = 0

rdmhyp1 = pd.DataFrame(index=labels, columns=labels) #setting up a hypothesis matrix with the x and y axes both being the list of 26 labels
rdmhyp1 = rdmhyp1.fillna(1) #fill the matrix with 1s
filter_col = [col for col in rdmhyp1 if col.startswith('Boun')]
#replace the 1s with 0s in all the cells whose columns start with "Boun"
for col in filter_col:
  for row in filter_col:
    if col[:6] == row[:6]:
      rdmhyp1.loc[row,col] = 0

rdmhyp = Adjacency(rdmhyp, matrix_type='distance', labels=labels) #calculate the distance between each cell in the matrix?
#hyprdm = rdmhyp.plot(cmap=sns.color_palette("rocket", as_cmap=True)) #plot this hypothesis RDM
#hyprdm.savefig(output_dir+'/hypothesisrdm_plot.png') #save the hypothesis RDM to a png file

rdmhyp1 = Adjacency(rdmhyp1, matrix_type='distance', labels=labels) #calculate the distance between each cell in the matrix?
#hyprdm1 = rdmhyp1.plot(cmap=sns.color_palette("rocket", as_cmap=True)) #plot this hypothesis RDM


all_sub_dissim={}; all_sub_comp_rsa={}; all_sub_comp1_rsa={}; dissim_dict={}
for roi in rois:
    print('Calculating '+roi+' dissimilarities')
    df = pd.DataFrame()
    for subj in subjs_list:
        print('doing sub '+subj)
        data_dir = os.path.join(bids_dir, 'fsloutput', 'preprocessed_'+subj+'.feat', 'stats', 'AHKJ_'+subj+'_1stGLM_40EVs_kids_newbounds.feat') # subject-level directories for t stats
        tstat_dir = os.path.join(data_dir, 'stats')

        file_list = glob.glob(os.path.join(tstat_dir, 'tstat*.nii.gz')) #pull all files from the tstat_dir beginning with 'tstat'
        file_list.sort() #sorting the tstat nifty files into numerical order
        #bringing in the order we actually want the files in since the tstat indices don't match up with the actual order
        fileorder = [0,11,22,33,35,36,37,38,39,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34]
        file_list = [file_list[i] for i in fileorder] #reordering and overwriting file list with the new order

        tstatlabels = [os.path.basename(x)[:-7].split('tstat_')[-1] for x in file_list] #creating labels from the list of tstats, removing full filepath and saving only, e.g. 'tstast16'

        roi_tstats = Brain_Data(file_list, mask='/data/projects/learning_lemurs/rsa/'+roi+'_thrbin_ped_common.nii.gz') #using the sub's tstats to generate an activation map within the roi mask 
    
        sub_pattern_dissim = roi_tstats.distance(metric='correlation') # create an individual subject dissimilarity matrix
        sub_pattern_dissim.labels = labels #add the labels imported earlier from the csv
        s_comp = sub_pattern_dissim.similarity(rdmhyp, metric='spearman', n_permute=0) #correlate this subject's neural RDM with the model/hypothesis RDM
        s_comp1 = sub_pattern_dissim.similarity(rdmhyp1, metric='spearman', n_permute=0) #correlate this subject's neural RDM with the model/hypothesis RDM

    
        all_sub_dissim[subj] = sub_pattern_dissim 
        #add each participant's correlation value to the dictionary
        all_sub_comp_rsa[subj] = s_comp['correlation']
        all_sub_comp1_rsa[subj] = s_comp1['correlation']   
 

        ########save full data out to be used in multilevel model########## 
        
        sub_pattern_dissim_square = squareform(sub_pattern_dissim.data, force='tomatrix', checks=True) #convert individual subject dissimilarity values to a 36x36 matrix
        sub_pattern_dissim_square_df = pd.DataFrame(sub_pattern_dissim_square, index=labels) #convert this matrix to a dfataframe
        tempsubdissim = pd.DataFrame(index=withbounlabels) #create a temporary data frame to hold the desired data, with short labels (one for each comparison rather than each window) as indices
        # select the values we want from the desired cells of the dataframe we just created for this participant
        chosenvalues = sub_pattern_dissim_square_df.iat[1,0], sub_pattern_dissim_square_df.iat[3,2], sub_pattern_dissim_square_df.iat[5,4], sub_pattern_dissim_square_df.iat[7,6], sub_pattern_dissim_square_df.iat[9,8], sub_pattern_dissim_square_df.iat[11,10], sub_pattern_dissim_square_df.iat[13,12], sub_pattern_dissim_square_df.iat[15,14], sub_pattern_dissim_square_df.iat[17,16], sub_pattern_dissim_square_df.iat[19,18], sub_pattern_dissim_square_df.iat[21,20], sub_pattern_dissim_square_df.iat[23,22], sub_pattern_dissim_square_df.iat[25,24], sub_pattern_dissim_square_df.iat[27,26], sub_pattern_dissim_square_df.iat[29,28], sub_pattern_dissim_square_df.iat[31,30], sub_pattern_dissim_square_df.iat[33,32], sub_pattern_dissim_square_df.iat[35,34], sub_pattern_dissim_square_df.iat[37,36], sub_pattern_dissim_square_df.iat[39,38]
        tempsubdissim['dissimvalues'] = chosenvalues # add the values we just selected to the df with the column name 'dissimvalues'
        tempsubdissim['pairtype'] = pairtypelabels # add the values we just selected to the df with the column name 'pairtype'
        tempsubdissim['subjnum'] = subj # add the subject number to all 18 cells of another column and label it 'subjnum'
        tempsubdissim['age'] = 'kid' # change this to correspond to whether we're using kids' or adults' boundaries
        tempsubdissim.reset_index(inplace=True) # Change the dataframe's index to a column
        df = df.append(tempsubdissim, ignore_index=True)  # Append each individual dataframe to one big dataframe called df
        
    df.to_csv(output_dir+'dissim_square_'+roi+'_kids_newbounds.csv') #save the one big dataframe to a csv 

        
    #print dissimilarity matrix plot for each ROI
    print('Making plot for '+roi)    
    all_sub_dissim_adj = [] # Create an empty list
    for subj in subjs_list:
        all_sub_dissim_adj.append(all_sub_dissim[subj].data) #add each subject's pattern dissimilarity to a single dissim_adj list

    all_sub_dissim_adj = np.asarray(all_sub_dissim_adj) # Turn list into a numpy array
     
    all_sub_dissim_mean = np.mean(all_sub_dissim_adj, axis=0) #calculate mean dissimilarity across all subjects      
    all_sub_dissim_mean_adj = all_sub_dissim[subj]
    all_sub_dissim_mean_adj.data = all_sub_dissim_mean  #repopulating it with the mean data for all subjects
    all_sub_dissim_mean_adj.plot(cmap=sns.color_palette("Reds_r", as_cmap=True))




    #label each comparison by whether it was shared between kids and adults or unique
    #this didn't end up being included in my final analysis for publication but useful for future reference

    sharedindices = [0,1,2,3,6,8]
    uniqueindices = [4,5,7,9]
    
    
    all_sub_dissim_array_shared = []
    all_sub_dissim_array_unique = []
    for subj in subjs_list:
        rdmhyp_array = np.array(rdmhyp.data)
        individ_sub_dissim_array = np.array(all_sub_dissim[subj].data)
        filtered_array = individ_sub_dissim_array * (rdmhyp_array != 1)
        filtered_array = filtered_array[filtered_array !=0]
        filtered_array_shared = [filtered_array[i] for i in sharedindices]
        filtered_array_unique = [filtered_array[i] for i in uniqueindices]
        all_sub_dissim_array_shared = np.append(all_sub_dissim_array_shared, filtered_array_shared, axis=0)  
        all_sub_dissim_array_unique = np.append(all_sub_dissim_array_unique, filtered_array_unique, axis=0)  
    all_sub_dissim_array_shared = np.asarray(all_sub_dissim_array_shared) 
    all_sub_dissim_array_unique = np.asarray(all_sub_dissim_array_unique) 
    np.savetxt('/data/projects/learning_lemurs/rsa/40EVs_kids/'+roi+'_within_data_kids_shared_new.csv', all_sub_dissim_array_shared, delimiter=",")
    np.savetxt('/data/projects/learning_lemurs/rsa/40EVs_kids/'+roi+'_within_data_kids_unique_new.csv', all_sub_dissim_array_unique, delimiter=",")
    

    all_sub_dissim_array_shared1 = []
    all_sub_dissim_array_unique1 = []
    for subj in subjs_list:
        rdmhyp1_array = np.array(rdmhyp1.data)
        individ_sub_dissim_array = np.array(all_sub_dissim[subj].data)
        filtered_array1 = individ_sub_dissim_array * (rdmhyp1_array != 1)
        filtered_array1 = filtered_array1[filtered_array1 !=0]
        filtered_array_shared1 = [filtered_array1[i] for i in sharedindices]
        filtered_array_unique1 = [filtered_array1[i] for i in uniqueindices]
        all_sub_dissim_array_shared1 = np.append(all_sub_dissim_array_shared1, filtered_array_shared1, axis=0)  
        all_sub_dissim_array_unique1 = np.append(all_sub_dissim_array_unique1, filtered_array_unique1, axis=0)  
    all_sub_dissim_array_shared1 = np.asarray(all_sub_dissim_array_shared1) 
    all_sub_dissim_array_unique1 = np.asarray(all_sub_dissim_array_unique1) 
    np.savetxt('/data/projects/learning_lemurs/rsa/40EVs_kids/'+roi+'_boundary_data_kids_shared_new.csv', all_sub_dissim_array_shared1, delimiter=",")
    np.savetxt('/data/projects/learning_lemurs/rsa/40EVs_kids/'+roi+'_boundary_data_kids_unique_new.csv', all_sub_dissim_array_unique1, delimiter=",")

        
        
    #If you DON't want it separated by shared/unique and DON'T want the full data for a MLM as above, you can just
    #save two columns of values instead - one within and one boundary - using the following code:
        
    #filter data array by hyp array for within to only include needed values and save to csv
    all_sub_dissim_array = []
    for subj in subjs_list:
        rdmhyp_array = np.array(rdmhyp.data)
        individ_sub_dissim_array = np.array(all_sub_dissim[subj].data)
        filtered_array = individ_sub_dissim_array * (rdmhyp_array != 1)
        filtered_array = filtered_array[filtered_array !=0]
        all_sub_dissim_array = np.append(all_sub_dissim_array, filtered_array, axis=0)  
    all_sub_dissim_array = np.asarray(all_sub_dissim_array) 
    np.savetxt(output_dir+roi+'_within_data_kids_new.csv', all_sub_dissim_array, delimiter=",")
        
        #filter data array by hyp array for boundary to only include needed values and save to csv
    all_sub_dissim_array1 = []
    for subj in subjs_list:
        rdmhyp1_array = np.array(rdmhyp1.data)
        individ_sub_dissim_array = np.array(all_sub_dissim[subj].data)
        filtered_array1 = individ_sub_dissim_array * (rdmhyp1_array != 1)
        filtered_array1 = filtered_array1[filtered_array1 !=0]
        all_sub_dissim_array1 = np.append(all_sub_dissim_array1, filtered_array1, axis=0)  
    all_sub_dissim_array1 = np.asarray(all_sub_dissim_array1)
    np.savetxt(output_dir+roi+'_boundary_data_kids_new.csv', all_sub_dissim_array1, delimiter=",")


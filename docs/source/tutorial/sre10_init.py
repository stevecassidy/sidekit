# -*- coding: utf-8 -*-
"""
Created on Tuesday Jul 14 2015

@author: Anthony Larcher

This script prepares all necessary lists to run a speaker recognition experiment 
on the NIST-SRE 2010 male extended task.

All available sphere files are listed, then compared to the I4U key file
in order to select all availale male sessions from NIST-SRE 04, 05, 06 and 08
as well as Switchboard (phon channel including telephone and microphone).

All sessions required to run the NIST-SRE 2010 male extended task are also listed.


HOW TO: run this script

    1) enter the number of parallel processes you want to run for the feature extraction

    2) change the corpora_dir dictionary to
       enter the directories where the NIST-SRE and Switchboard data are stored
       so that the script can look through your directories.
       
    3) enter the feature_root_dir where you want to store your MFCC files.
       MFCC will be stored under the directories: 
       
       feature_root_dir/sre04
       feature_root_dir/sre05
       feature_root_dir/sre06
       ... (and so on)
       
       All MFCCs will be stored in one single directory per database, one MFCC file per SPHERE file and per channel
    
"""


import sys
import os
import random
import numpy as np
import pandas as pd
pd.set_option('display.mpl_style', 'default')
from fnmatch import fnmatch
from collections import namedtuple
import pickle


## TO MODIFIY to run
nbThread = 20  # define here the number of parallel process you want to run (for feature extraction).

corpora_dir = {'sre05': '/lium/paroleh/NIST-SRE/LDC2009E100/SRE05',
               'sre04': '/lium/paroleh/NIST-SRE/LDC2009E100/SRE04',
               'sre06': '/lium/paroleh/NIST-SRE/LDC2009E100/SRE06',
               'sre08': '/lium/paroleh/NIST-SRE/LDC2009E100/SRE08',
               'sre10': '/lium/paroleh/NIST-SRE/LDC2012E09/SRE10',
               'swb': ['/lium/paroleh/NIST-SRE/Switchboard-1-LDC2001S13',
                       '/lium/paroleh/NIST-SRE/Switchboard-2P2-LDC99S79',
                       '/lium/paroleh/NIST-SRE/Switchboard-2P3-LDC2002S06',
                       '/lium/paroleh/NIST-SRE/Switchboard-Cell-P2-LDC2004S07']}
               }

feature_root_dir = '/lium/parolee/larcher/data/nist'
##




Selection = namedtuple("Selection", "gender db speechType channelType min_duration max_duration, min_session_nb")


corpora = {'MIX04': 'sre04',
        'MIX05': 'sre05',
        'MIX06': 'sre06',
        'MIX08': 'sre08',
        'MIX10': 'sre10',
        'SWCELLP1': 'swb',
        'SWCELLP2': 'swb',
        'SWPH1': 'swb',
        'SWPH2': 'swb',
        'SWPH3': 'swb',        
        }

def search_files(corpora_dir, extension):
    """"""
    corpusList = []
    completeFileList = []
    fileList = []
    file_dict = {}
    for corpus in corpora_dir.keys():
        if not isinstance(corpora_dir[corpus], list):
            dirNameList = [corpora_dir[corpus]]
        else:
            dirNameList = corpora_dir[corpus]
        for dirName in dirNameList:
            print("Scanning {}\n".format(dirName))
            for path, subdirs, files in os.walk(dirName):
                for name in files:
                    if fnmatch(name, extension.upper()) or fnmatch(name, extension.lower()):
                        name = os.path.splitext(name)[0]
                        file_dict[corpus + '/' + os.path.splitext(name)[0].lower()] = os.path.join(path, name)
                        corpusList.append(corpus)
                        completeFileList.append(os.path.join(path, name))
                        fileList.append((corpus + '/' + os.path.splitext(name)[0]).lower())
    return corpusList, completeFileList, fileList, file_dict


extension = '*.sph'
corpusList, completeFileList, sphList, file_dict = search_files(corpora_dir, extension)
with open('nist_existing_sph_files.p', "wb" ) as f:
            pickle.dump( (corpusList, completeFileList, sphList), f)

print("After listing, {} files found\n".format(len(completeFileList)))

######################################################################
# Load the list of files to process for evaluation (enrollment and test)
######################################################################
trn_male = sidekit.IdMap('task/original_sre10_coreX-coreX_m_trn.h5')
ndx_male = sidekit.Ndx('task/original_sre10_coreX-coreX_m_ndx.h5')

sre10_male_sessions = np.unique(np.concatenate((trn_male.rightids, ndx_male.segset), axis=1))


######################################################################
# List now all the files we would like to process for training purposes
######################################################################
"""
    COLUMNS of the I4U.key
    
    0 database
    1 speaker
    2 session
    3 channel
    4 gender
    5 birth
    6 recordYear
    7 age
    8 speechType
    9 channelType
    10 TIM
    11 length
    12 lang
    13 nativeLang
    14 effort
    """


# Load dataframe
i4u_df = pd.read_csv('Sph_MetaData/I4U.key', low_memory=False)

# Create keys corresponding to NIST info
i4u_df.database.replace(corpora.keys(), corpora.values(), inplace=True)
i4u_df["filename"] = np.nan
i4u_df["nistkey"] = i4u_df.database + '/' + i4u_df.session

i4u_df.channel.replace(['a', 'b', 'x'], ['_a', '_b', ''], inplace=True)
i4u_df["sessionKey"] = i4u_df.nistkey + i4u_df.channel

# Set selection criteria
select = Selection(gender=['m'], 
                   db= ['swb', 'sre04', 'sre05', 'sre06', 'sre08'],
                   speechType=['tel', 'mic'],
                   channelType=['phn'],
                   min_duration=30, 
                   max_duration=3000,
                   min_session_nb=1)



# select IDEAL list of sessions to keep
keep_sessions = i4u_df[(i4u_df.database.isin(select.db) \
                     &  i4u_df.gender.isin(select.gender) \
                     & i4u_df.speechType.isin(select.speechType) \
                     & i4u_df.channelType.isin(select.channelType) \
                     & (i4u_df.length <= select.max_duration) \
                     & (i4u_df.length >= select.min_duration)) \
                     | i4u_df.sessionKey.isin(sre10_male_sessions)]
          
# Select speakers with enough sessions
spk_count = keep_sessions['speaker'].value_counts(normalize=False, 
                                    sort=True, ascending=False, bins=None)
spk_id = np.array(spk_count.index.tolist())
spk_keep = spk_id[spk_count.get_values() > select.min_session_nb]           

keep_sessions = keep_sessions[keep_sessions['speaker'].isin(spk_keep) \
                               | keep_sessions['database'].isin(['sre10'])]

print(('Keep {} sessions from {} speakers'.format(keep_sessions.shape[0], 
                               keep_sessions.speaker.unique().shape[0])))


######################################################################
# from the existing files, extract the list of files we want to keep
# create 2 lists: audio_file_dir and feature_file_list
# input and output for the feature extraction method
######################################################################
keep_sessions = keep_sessions[keep_sessions.nistkey.isin(sphList)]
for nk in keep_sessions.nistkey:
    keep_sessions.filename[keep_sessions.nistkey == nk] = file_dict[nk]

audio_file_list = keep_sessions.filename.as_matrix()
unique_idx = np.unique(audio_file_list, return_index=True)
audio_file_list = audio_file_list[unique_idx[1]]
feature_file_list = keep_sessions.nistkey.as_matrix()[unique_idx[1]]

with open('sph_files_to_process.p', "wb" ) as f:
    pickle.dump( (audio_file_list, feature_file_list), f)


print("Found {} sphere files to process\n".format(feature_file_list.shape[0]))


####################################################################
# Extract feature files in a single directory per database.
# After extraction, a list of the existing feature files is created
####################################################################
fs = sidekit.FeaturesServer(input_dir='',
                 input_file_extension='.sph',
                 label_dir='./',
                 label_file_extension='.lbl',
                 from_file='audio',
                 config='sid_8k')

# Shuffle the features to harmonize the length of each process (files from some years 
# are longer than others then shuffling optimizes the use of resources)
idx = np.arange(len(audio_file_list))
random.shuffle(idx)
audio_file_list = audio_file_list[idx]
feature_file_list = feature_file_list[idx]

fs.save_parallel(audio_file_list, feature_file_list, 'spro4', feature_root_dir,
                         '.mfcc', and_label=False, numThread=nbThread)


####################################################################
# Create a new list with the existing feature files 
####################################################################
feature_dir = {'sre05': feature_root_dir + '/sre05',
               'sre04': feature_root_dir + '/sre04',
               'sre06': feature_root_dir + '/sre06',
               'sre08': feature_root_dir + '/sre08',
               'sre10': feature_root_dir + '/sre10',
               'swb': feature_root_dir + '/swb'}
existingFeatureList = search_files(feature_dir, '*.mfcc')[2]

keep_sessions["featureExist"] = keep_sessions['sessionKey'].isin(existingFeatureList)

if not keep_sessions["featureExist"].sum() == len(keep_sessions):
    print('After feature extraction, {} sessions are missing'.format(len(keep_sessions) - keep_sessions["featureExist"].sum()))

# After a first extraction, list all missing features and try again to extract.
# This is done in case one of the thread break for any possible reason.
audio_file_list = keep_sessions[~keep_sessions["featureExist"]].filename.as_matrix()
feature_file_list = keep_sessions[~keep_sessions["featureExist"]].nistkey.as_matrix()

fs.save_parallel(audio_file_list, feature_file_list, 'spro4', feature_root_dir,
                         '.mfcc', and_label=False, numThread=nbThread)

if not keep_sessions["featureExist"].sum() == len(keep_sessions):
    print('After feature extraction, {} sessions are missing'.format(len(keep_sessions) - keep_sessions["featureExist"].sum()))

existingFeatureList = search_files(feature_dir, '*.mfcc')[2]

keep_sessions["featureExist"] = keep_sessions['sessionKey'].isin(existingFeatureList)

####################################################################
# Associate each feature file with the speaker involved and the sessionKey
####################################################################
train_sessions = keep_sessions[keep_sessions['sessionKey'].isin(existingFeatureList) \
               & ~keep_sessions['database'].isin(['sre10'])]

idmap_sre04050608_male = sidekit.IdMap()
idmap_sre04050608_male.leftids = np.array(train_sessions.speaker)
idmap_sre04050608_male.rightids = np.array(train_sessions.sessionKey)
idmap_sre04050608_male.start = np.empty(idmap_sre04050608_male.leftids.shape, '|O')
idmap_sre04050608_male.stop = np.empty(idmap_sre04050608_male.leftids.shape, '|O')
idmap_sre04050608_male.validate()
idmap_sre04050608_male.save('task/sre04050608_m_training.h5')

print('Save the background training IdMap with\n   {} sessions from\n   {} speakers'.format(idmap_sre04050608_male.leftids.shape[0], np.unique(idmap_sre04050608_male.leftids).shape[0]))

####################################################################
# Create a list to train the UBM by randomly selecting
# 500 sessions
####################################################################
print('Create the training list for the UBM')
ubm_list = random.sample(idmap_sre04050608_male.rightids, 500)
with open('task/ubm_list.txt','w') as of:
    of.write("\n".join(ubm_list))


############################################################
# Create an IdMap to compute the test segment statistics
############################################################
print('Create the IdMap for the test segments')
test_idmap = sidekit.IdMap()
# Remove missing files from the test data
existingTestSeg, segs = sidekit.sv_utils.check_file_list(ndx_male.segset,
                                feature_root_dir, '.mfcc')
test_idmap.rightids = ndx_male.segset[segs]
test_idmap.leftids = ndx_male.segset[segs]
test_idmap.start = np.empty(test_idmap.rightids.shape, '|O')
test_idmap.stop = np.empty(test_idmap.rightids.shape, '|O')
test_idmap.validate()
test_idmap.save('task/sre10_coreX-coreX_m_test.h5')

#############################################################
# Remove missing files for enrolment
#############################################################
existingTestSeg, segs = sidekit.sv_utils.check_file_list(trn_male.rightids,
                                feature_root_dir, '.mfcc')
trn_male.rightids = trn_male.rightids[segs]
trn_male.leftids = trn_male.leftids[segs]
trn_male.start = np.empty(trn_male.rightids.shape, '|O')
trn_male.stop = np.empty(trn_male.rightids.shape, '|O')
trn_male.validate()
trn_male.save('task/sre10_coreX-coreX_m_trn.h5')

#############################################################
# Remove missing models and segments from the Ndx
#############################################################
existingTestSeg, segs = sidekit.sv_utils.check_file_list(ndx_male.segset,
                                feature_root_dir, '.mfcc')
ndx_male = ndx_male.filter(trn_male.leftids, existingTestSeg, keep=True)
ndx_male.save('task/sre10_coreX-coreX_m_ndx.h5')







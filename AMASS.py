#!/usr/bin/env python3
# -*- coding: utf-8 -*-

amass_splits = {
    'train': [
        'ACCAD',
        'CMU',
        'EKUT',
        'Eyes_Japan_Dataset',
        'KIT',
        'MPI_Limits',
        'TotalCapture',
        'TCD_handMocap',
    ],
    'test': [
        'BioMotionLab_NTroje',
    ],
}

import os
import os.path as osp
from tqdm import tqdm

import numpy as np
from dataset.motion_preprocessing import smpl_to_generalized_coordinates

# %%
def read_data(trunk_datafolder, trunk_path, folder, sequences, seqlen, overlap=0.75, fps=30):

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        read_sequence(trunk_datafolder, trunk_path, seq_folder, seq_name, fps, seqlen, overlap)

    return trunk_path

def read_sequence(trunk_datafolder, trunk_path, folder, seq_name, fps, seqlen, overlap):
    
    subjects = sorted(list(filter(lambda x: osp.isdir(os.path.join(folder, x)), os.listdir(folder))))
    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            
            if fname.endswith('shape.npz'):
                continue

            data = np.load(fname)
            mocap_framerate = int(data['mocap_framerate'])
            try:
                gender = str(data['gender']).decode()
            except:
                gender = str(data['gender'])

            if len(gender)==4:
                gender_id = 0
            if len(gender)==6 or len(gender)==9:
                gender_id = 1
            
            # pose       
            sampling_freq = mocap_framerate // fps
            pose = data['poses'][::sampling_freq]
            if len(pose) < 80:
                continue

            q = smpl_to_generalized_coordinates(pose, data['trans'][::sampling_freq])

            # shape and gender
            shape = np.repeat(data['betas'][:10][np.newaxis], q.shape[0], axis=0)
            gender_id = np.repeat(np.array([[gender_id]]), q.shape[0], axis=0)
            
            data_i = np.concatenate([q, shape, gender_id], axis=1)

            vid_name = f'{seq_name}_{subject}_{action[:-4]}'

            n_frames = shape.shape[0]
            for frame in range(0, n_frames-seqlen, int(seqlen * (1-overlap))):
                trunk_path_f = trunk_datafolder + '/%s_vid_%d' % (vid_name, frame)
                np.save(trunk_path_f, data_i[frame:frame+seqlen])
                trunk_path.append(trunk_path_f)
    return


# %%
import pickle

amass_dir = './dataset/data_raw/AMASS'
save_dir = './dataset/data_processed'
seqlen = 50

for split in ['test', 'train']:
    if split == 'train':
        overlap = 9/10
    else:
        overlap = 0
    trunk_datafolder = osp.join(amass_dir, split, str(seqlen))
    if not osp.exists(trunk_datafolder):
        os.makedirs(trunk_datafolder)
        
    trunk_path = []
    data = read_data(trunk_datafolder, trunk_path, amass_dir, sequences=amass_splits[split], seqlen=seqlen, overlap=overlap)

    data_file = osp.join(save_dir, f'amass_{split}_%d.pkl' % seqlen)
    
    print(f'Saving AMASS dataset to {data_file} with total number of data %d' % len(data))
    pickle.dump(data, open(data_file, 'wb'))



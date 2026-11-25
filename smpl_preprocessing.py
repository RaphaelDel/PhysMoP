import argparse
import numpy as np
import pickle
import scipy.io as sio

# This code is used to 
# 1. convert the chumpy object of the official model into numpy array
# 2. add different joint regressor to the processed model

# official model path
src_path = './basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
# output model path
output_path = './processed_' + src_path.split('/')[-1]

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess SMPL model')
    parser.add_argument('--src', type=str, default=src_path)
    parser.add_argument('--output', type=str, default=output_path)
    args = parser.parse_args()

    # joint regressor path
    human36_path = './regressor/J_regressor_h36m.npy'
    coco_path = './regressor/J_regressor_cocoplus.npy'
    extra_path = './regressor/J_regressor_extra.npy'

    human36 = ['Pelvis', 'RHip', 'RKnee','RAnkle','LHip','LKnee','LAnkle',
            'Spine','Thorax','Neck','Head', 
            'LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist']
    coco = ['RAnkle', 'RKnee', 'RHip', 'LHip', 'LKnee', 'LAnkle', 
            'RWrist', 'RElbow', 'RShoulder', 'LShoulder', 'LElbow', 'LWrist',
            'Nose', 'LEye', 'REye', 'LEar', 'REar']
    smpl = ['Pelvis', 'LHip', 'RHip', 'Spine3', 'LKnee', 'RKnee', 'Spine6',
            'LAnkle', 'RAnkle', 'Spine9', 'LFoot', 'RFoot', 'Thorax', 'Spine13', 'Spine14',
            'Neck', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHand', 'RHand']

    parent_human36 = np.asarray([7,0,1,2,0,4,5,8,9,10,10,8,11,12,8,14,15])
    parent_coco_plus = np.asarray([0,0,0,1,2]) + 17
    parent_extra = np.asarray([10])

    JOINT_PARENT = np.concatenate([parent_human36,parent_coco_plus,parent_coco_plus])

    J_regressor_h36m = np.load(human36_path)
    J_regressor_extra = np.load('./dataset/smpl_official/regressor/J_regressor_extra.npy')
    J_regressor_cocoplus = np.load(coco_path, allow_pickle=True)

    J_regressor = np.concatenate([J_regressor_h36m,J_regressor_cocoplus[-5:,:],J_regressor_extra[3:4,:]], axis=0)
    J_regressor_h36m_coco = [[11,12,13,14,15,16, 8,10],
                             [ 9,10,11, 8, 7, 6,12,13]]
    J_regressor[J_regressor_h36m_coco[0]] = (J_regressor_h36m[J_regressor_h36m_coco[0]]+J_regressor_cocoplus[J_regressor_h36m_coco[1]]) / 2
    J_regressor_h36m_extra = [[7,8,10],
                             [6,2,8]]
    J_regressor[J_regressor_h36m_extra[0]] = (J_regressor_h36m[J_regressor_h36m_extra[0]]+J_regressor_extra[J_regressor_h36m_extra[1]]) / 2

    with open(args.src, 'rb') as f:
        src_data = pickle.load(f, encoding="latin1")
        src_data['kintree_table'][0][0] = 0
        v_template = np.array(src_data['v_template'])
        model = {
            'V_regressor': src_data['J_regressor'].todense(),
            'J_regressor': J_regressor,
            'weights': np.array(src_data['weights']),
            'posedirs': np.array(src_data['posedirs']),
            'v_template': v_template,
            'shapedirs': np.array(src_data['shapedirs']),
            'f': np.array(src_data['f']),
            'kintree_table_J': JOINT_PARENT,
            'kintree_table': src_data['kintree_table'][0],
            }

    
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
        
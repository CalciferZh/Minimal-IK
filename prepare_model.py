from config import *
import pickle
import numpy as np


def prepare_mano_model():
  """
  Convert the official MANO model into compatible format with this project.
  """
  with open(OFFICIAL_MANO_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
  params = {
    'pose_pca_basis': np.array(data['hands_components']),
    'pose_pca_mean': np.array(data['hands_mean']),
    'J_regressor': data['J_regressor'].toarray(),
    'skinning_weights': np.array(data['weights']),
    # pose blend shape
    'mesh_pose_basis': np.array(data['posedirs']),
    'mesh_shape_basis': np.array(data['shapedirs']),
    'mesh_template': np.array(data['v_template']),
    'faces': np.array(data['f']),
    'parents': data['kintree_table'][0].tolist(),
  }
  params['parents'][0] = None
  with open(MANO_MODEL_PATH, 'wb') as f:
    pickle.dump(params, f)


def prepare_smpl_model():
  """
  Convert the official SMPL model into compatible format with this project.
  """
  with open(OFFICIAL_SMPL_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
  params = {
    # SMPL does not provide pose PCA
    'pose_pca_basis': np.eye(23 * 3),
    'pose_pca_mean': np.zeros(23 * 3),
    'J_regressor': data['J_regressor'].toarray(),
    'skinning_weights': np.array(data['weights']),
    # pose blend shape
    'mesh_pose_basis': np.array(data['posedirs']),
    'mesh_shape_basis': np.array(data['shapedirs']),
    'mesh_template': np.array(data['v_template']),
    'faces': np.array(data['f']),
    'parents': data['kintree_table'][0].tolist(),
  }
  params['parents'][0] = None
  with open(SMPL_MODEL_PATH, 'wb') as f:
    pickle.dump(params, f)


def prepare_smplh_model():
  """
  Convert the official SMPLH model into compatible format with this project.
  """
  data = np.load(OFFICIAL_SMPLH_PATH)
  params = {
    # SMPL does not provide pose PCA
    'pose_pca_basis': np.eye(51 * 3),
    'pose_pca_mean': np.zeros(51 * 3),
    'J_regressor': data['J_regressor'],
    'skinning_weights': np.array(data['weights']),
    # pose blend shape
    'mesh_pose_basis': np.array(data['posedirs']),
    'mesh_shape_basis': np.array(data['shapedirs']),
    'mesh_template': np.array(data['v_template']),
    'faces': np.array(data['f']),
    'parents': data['kintree_table'][0].tolist(),
  }
  params['parents'][0] = None
  with open(SMPLH_MODEL_PATH, 'wb') as f:
    pickle.dump(params, f)


if __name__ == '__main__':
  prepare_smplh_model()

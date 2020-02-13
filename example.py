from solver import *
from armatures import *
from models import *
import numpy as np
import config


np.random.seed(20160923)
pose_glb = np.zeros([1, 3]) # global rotation


########################## mano settings #########################
n_pose = 12 # number of pose pca coefficients, in mano the maximum is 45
n_shape = 10 # number of shape pca coefficients
pose_pca = np.random.normal(size=n_pose)
shape = np.random.normal(size=n_shape)
mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=1000)


########################## smpl settings ##########################
# note that in smpl and smpl-h no pca for pose is provided
# therefore in the model we fake an identity matrix as the pca coefficients
# to make the code compatible

# n_pose = 23 * 3 # degrees of freedom, (n_joints - 1) * 3
# n_shape = 10
# pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
# shape = np.random.normal(size=n_shape)
# mesh = KinematicModel(config.SMPL_MODEL_PATH, SMPLArmature, scale=10)


########################## smpl-h settings ##########################
# n_pose = 51 * 3
# n_shape = 16
# pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
# shape = np.random.normal(size=n_shape)
# mesh = KinematicModel(config.SMPLH_MODEL_PATH, SMPLHArmature, scale=10)


########################## solving example ############################

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = Solver(verbose=True)

_, keypoints = \
  mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
params_est = solver.solve(wrapper, keypoints)

shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

print('----------------------------------------------------------------------')
print('ground truth parameters')
print('pose pca coefficients:', pose_pca)
print('pose global rotation:', pose_glb)
print('shape: pca coefficients:', shape)

print('----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)

mesh.set_params(pose_pca=pose_pca)
mesh.save_obj('./gt.obj')
mesh.set_params(pose_pca=pose_pca_est)
mesh.save_obj('./est.obj')

print('ground truth and estimated meshes are saved into gt.obj and est.obj')

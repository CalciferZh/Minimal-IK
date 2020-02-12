from solver import *
from armatures import *
from models import *
import numpy as np
import config

n_shape = 10
np.random.seed(20160923)

########################## mano settings #########################
# n_pose = 12
# n_shape = 10
# np.random.seed(20160923)
# pose_pca = np.random.normal(size=n_pose)
# pose_glb = np.zeros([1, 3])
# shape = np.random.normal(size=n_shape)
# mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=1000)


########################## smpl settings ##########################
n_pose = 23 * 3
pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
pose_glb = np.zeros([1, 3])
shape = np.random.normal(size=n_shape)
mesh = KinematicModel(config.SMPL_MODEL_PATH, SMPLArmature, scale=10)


########################## solving example ############################

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = Solver(verbose=True)

_, keypoints = \
  mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
params_est = solver.solve(wrapper, keypoints)

shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

print('-----------------------------------------------------------------------')
print('ground truth parameters')
print('pose pca coefficients:', pose_pca)
print('pose global rotation:', pose_glb)
print('shape: pca coefficients:', shape)

print('-----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)

mesh.set_params(pose_pca=pose_pca)
mesh.save_obj('./gt.obj')
mesh.set_params(pose_pca=pose_pca_est)
mesh.save_obj('./est.obj')

print('ground truth and estimated meshes are saved into gt.obj and est.obj')

from solver import Solver
from armatures import ManoArmature
from models import KinematicModel
from models import KinematicPCAWrapper
from vctoolkit import obj_save
import numpy as np
import config

n_pose = 12
n_shape = 10
np.random.seed(20160923)
pose_pca = np.random.normal(size=n_pose)
pose_glb = np.zeros([1, 3])
shape = np.random.normal(size=n_shape)

mesh = KinematicModel(config.MANO_MODEL_PATH, ManoArmature)
wrapper = KinematicPCAWrapper(mesh, n_pose=12)
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

verts, _ = mesh.set_params(pose_pca=pose_pca)
obj_save('./gt.obj', verts, mesh.faces)
verts, _ = mesh.set_params(pose_pca=pose_pca_est)
obj_save('./est.obj', verts, mesh.faces)

print('ground truth and estimated meshes are saved into gt.obj and est.obj')

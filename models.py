import numpy as np
import pickle


class KinematicModel():
  """
  Kinematic model that takes in model parameters and outputs mesh, keypoints,
  etc.
  """
  def __init__(self, model_path, armature, scale=1):
    """
    Parameters
    ----------
    model_path : str
      Path to the model to be loaded.
    armature : object
      An armature class from `armatures.py`.
    scale : int, optional
      Scale of the model to make the solving easier, by default 1
    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.pose_pca_basis = params['pose_pca_basis']
      self.pose_pca_mean = params['pose_pca_mean']

      self.J_regressor = params['J_regressor']

      self.skinning_weights = params['skinning_weights']

      self.mesh_pose_basis = params['mesh_pose_basis'] # pose blend shape
      self.mesh_shape_basis = params['mesh_shape_basis']
      self.mesh_template = params['mesh_template']

      self.faces =  params['faces']

      self.parents = params['parents']

    self.n_shape_params = self.mesh_shape_basis.shape[-1]
    self.scale = scale

    self.armature = armature
    self.n_joints = self.armature.n_joints
    self.pose = np.zeros((self.n_joints, 3))
    self.shape = np.zeros(self.mesh_shape_basis.shape[-1])
    self.verts = None
    self.J = None
    self.R = None
    self.keypoints = None

    self.J_regressor_ext = \
      np.zeros([self.armature.n_keypoints, self.J_regressor.shape[1]])
    self.J_regressor_ext[:self.armature.n_joints] = self.J_regressor
    for i, v in enumerate(self.armature.keypoints_ext):
      self.J_regressor_ext[i + self.armature.n_joints, v] = 1

    self.update()

  def set_params(self, pose_abs=None, pose_pca=None, pose_glb=None, shape=None):
    """
    Set model parameters and get the mesh. Do not set `pose_abs` and `pose_pca`
    at the same time.

    Parameters
    ----------
    pose_abs : np.ndarray, shape [n_joints, 3], optional
      The absolute model pose in axis-angle, by default None
    pose_pca : np.ndarray, optional
      The PCA coefficients of the pose, shape [n_pose, 3], by default None
    pose_glb : np.ndarray, shape [1, 3], optional
      Global rotation for the model, by default None
    shape : np.ndarray, shape [n_shape], optional
      Shape coefficients of the pose, by default None

    Returns
    -------
    np.ndarray, shape [N, 3]
      Vertices coordinates of the mesh, scale applied.
    np.ndarray, shape [K, 3]
      Keypoints coordinates of the model, scale applied.
    """
    if pose_abs is not None:
      self.pose = pose_abs
    elif pose_pca is not None:
      self.pose = np.dot(
        np.expand_dims(pose_pca, 0), self.pose_pca_basis[:pose_pca.shape[0]]
      )[0] + self.pose_pca_mean
      self.pose = np.reshape(self.pose, [self.n_joints - 1, 3])
      if pose_glb is None:
        pose_glb = np.zeros([1, 3])
      pose_glb = np.reshape(pose_glb, [1, 3])
      self.pose = np.concatenate([pose_glb, self.pose], 0)
    if shape is not None:
      self.shape = shape
    return self.update()

  def update(self):
    """
    Re-compute vertices and keypoints with given parameters.

    Returns
    -------
    np.ndarray, shape [N, 3]
      Vertices coordinates of the mesh, scale applied.
    np.ndarray, shape [K, 3]
      Keypoints coordinates of the model, scale applied.
    """
    verts = self.mesh_template + self.mesh_shape_basis.dot(self.shape)
    self.J = self.J_regressor.dot(verts)
    self.R = self.rodrigues(self.pose.reshape((-1, 1, 3)))
    G = np.empty((self.n_joints, 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.n_joints):
      G[i] = G[self.parents[i]].dot(self.with_zeros(
          np.hstack([
            self.R[i],
            (self.J[i, :] - self.J[self.parents[i], :]).reshape([3, 1])
          ])
      ))
    G = G - self.pack(np.matmul(
        G,
        np.hstack([self.J, np.zeros([self.n_joints, 1])]) \
          .reshape([self.n_joints, 4, 1])
    ))
    T = np.tensordot(self.skinning_weights, G, axes=[[1], [0]])
    verts = np.hstack((verts, np.ones([verts.shape[0], 1])))

    self.verts = \
      np.matmul(T, verts.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.keypoints = self.J_regressor_ext.dot(self.verts)

    self.verts *= self.scale
    self.keypoints *= self.scale

    return self.verts.copy(), self.keypoints.copy()

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].
    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0), [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_obj(self, path):
    """
    Save the SMPL model into .obj file.
    Parameter:
    ---------
    path: Path to save.
    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class KinematicPCAWrapper():
  """
  A wrapper for `KinematicsModel` to be compatible to the solver.
  """
  def __init__(self, core, n_pose=12):
    """
    Parameters
    ----------
    core : KinematicModel
      Core model to be manipulated.
    n_pose : int, optional
      Degrees of freedom for pose, by default 12
    """
    self.core = core
    self.n_pose = n_pose
    self.n_shape = core.n_shape_params
    self.n_glb = 3
    self.n_params = self.n_pose + self.n_shape + self.n_glb

  def run(self, params):
    """
    Set the parameters, return the corresponding result.

    Parameters
    ----------
    params : np.ndarray
      Model parameters.

    Returns
    -------
    np.ndarray
      Corresponding result.
    """
    shape, pose_pca, pose_glb = self.decode(params)
    return \
      self.core.set_params(pose_glb=pose_glb, pose_pca=pose_pca, shape=shape)[1]

  def decode(self, params):
    """
    Decode the compact model parameters into semantic parameters.

    Parameters
    ----------
    params : np.ndarray
      Model parameters.

    Returns
    -------
    np.ndarray
      Shape parameters.
    np.ndarray
      Pose parameters.
    np.ndarray
      Global rotation.
    """
    pose_glb = params[:self.n_glb]
    pose_pca = params[self.n_glb:-self.n_shape]
    shape = params[-self.n_shape:]
    return shape, pose_pca, pose_glb

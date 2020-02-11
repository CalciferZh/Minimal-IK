from tqdm import tqdm
import numpy as np
from vctoolkit import Timer


class Solver:
  def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
    self.eps = eps
    self.max_iter = max_iter
    self.mse_threshold = mse_threshold
    self.verbose = verbose
    self.timer = Timer()

  def get_derivative(self, model, params, n):
    params1 = np.array(params)
    params2 = np.array(params)

    params1[n] += self.eps
    params2[n] -= self.eps

    res1 = model.run(params1)
    res2 = model.run(params2)

    d = (res1 - res2) / (2 * self.eps)

    return d.ravel()

  def solve(self, model, target, init=None, u=1e-3, v=1.5):
    if init is None:
      init = np.zeros(model.n_params)
    out_n = np.shape(model.run(init).ravel())[0]
    jacobian = np.zeros([out_n, init.shape[0]])

    last_update = 0
    last_mse = 0
    params = init
    for i in range(self.max_iter):
      residual = (model.run(params) - target).reshape(out_n, 1)
      mse = np.mean(np.square(residual))

      if abs(mse - last_mse) < self.mse_threshold:
        return params

      for k in range(params.shape[0]):
        jacobian[:, k] = self.get_derivative(model, params, k)

      jtj = np.matmul(jacobian.T, jacobian)
      jtj = jtj + u * np.eye(jtj.shape[0])

      update = last_mse - mse
      delta = np.matmul(
        np.matmul(np.linalg.inv(jtj), jacobian.T), residual
      ).ravel()
      params -= delta

      if update > last_update and update > 0:
        u /= v
      else:
        u *= v

      last_update = update
      last_mse = mse

      if self.verbose:
        print(i, self.timer.tic(), mse)

    return params

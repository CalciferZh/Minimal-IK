from tqdm import tqdm
import numpy as np
from vctoolkit import Timer


class Solver:
  def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
    """
    Parameters
    ----------
    eps : float, optional
      Epsilon for derivative computation, by default 1e-5
    max_iter : int, optional
      Max iterations, by default 30
    mse_threshold : float, optional
      Early top when mse change is smaller than this threshold, by default 1e-8
    verbose : bool, optional
      Print information in each iteration, by default False
    """
    self.eps = eps
    self.max_iter = max_iter
    self.mse_threshold = mse_threshold
    self.verbose = verbose
    self.timer = Timer()

  def get_derivative(self, model, params, n):
    """
    Compute the derivative by adding and subtracting epsilon

    Parameters
    ----------
    model : object
      Model wrapper to be manipulated.
    params : np.ndarray
      Current model parameters.
    n : int
      The index of parameter.

    Returns
    -------
    np.ndarray
      Derivative with respect to the n-th parameter.
    """
    params1 = np.array(params)
    params2 = np.array(params)

    params1[n] += self.eps
    params2[n] -= self.eps

    res1 = model.run(params1)
    res2 = model.run(params2)

    d = (res1 - res2) / (2 * self.eps)

    return d.ravel()

  def solve(self, model, target, init=None, u=1e-3, v=1.5):
    """
    Solver for the target.

    Parameters
    ----------
    model : object
      Wrapper to be manipulated.
    target : np.ndarray
      Optimization target.
    init : np,ndarray, optional
      Initial parameters, by default None
    u : float, optional
      LM algorithm parameter, by default 1e-3
    v : float, optional
      LM algorithm parameter, by default 1.5

    Returns
    -------
    np.ndarray
      Solved model parameters.
    """
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

# Minimal-IK

A simple and naive inverse kinematics solver for MANO hand model, SMPL body model, and SMPLH body+hand model.

Briefly, given joint coordinates (and optional other keypoints), the solver gives the corresponding model parameters.

Levenberg–Marquardt algorithm is used, the energy is simply the L2 distance between the keypoints.

## Usage

### Models

1. Download the official model from MPI.
2. See `config.py` and set the official model path.
3. See `prepare_model.py`, use the provided function to pre-process the model.

### Solver

1. See `example.py`, un-comment the corresponding code.
2. `python example.py`.
3. The example ground truth mesh and estimated mesh are saved to `gt.obj` and `est.obj` respectively.

### Dependencies

Every required package is available via `pip install`.

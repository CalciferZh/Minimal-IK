# Minimal-IK

A simple and naive inverse kinematics solver for MANO hand model, SMPL body model, and SMPL-H body+hand model.

Briefly, given joint coordinates (and optional other keypoints), the solver gives the corresponding model parameters.

Levenberg–Marquardt algorithm is used, the energy is simply the L2 distance between the keypoints.

## Results

### Qualitative

This is the example result on the SMPL body model.
The left is the ground truth, and the right one is the estimation.
You can notice the minor difference between the right hands.

![](body.png)

Below is the example result of the MANO hand model.
Left for ground truth, and right for estimation.

![](hand.png)

### Quantitative

We test this approach on the [AMASS dataset](https://amass.is.tue.mpg.de/).

|             | Mean Joint Error (mm) | Mean Vertex Error (mm) |
| ----------  | --------------------- | ---------------------- |
| SMPL (body) | 8.717                 | 14.136                 |


In the test on AMASS, we assume that the global rotation is known.
This is because this optimization based approach cannot handle large global rotations.

(We'll update the hand results soon.)

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

### Limitations

Again, we note that this approach cannot handle large global rotations (R0) due to the high non-convexity.
For example, when the subject keeps the T pose but faces backwards.

In such cases, a good initialization, at least for R0, is necessary.

## Credits

* @yxyyyxxyy for the quantitative test on the AMASS dataset.
* @zjykljf for the starter code of the LM solver.

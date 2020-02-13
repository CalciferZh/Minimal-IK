# it must be assured that parent joint appears before child joint

class MANOArmature:
  n_joints = 16

  # indices of extended keypoints
  keypoints_ext = [333, 444, 672, 555, 744]

  n_keypoints = n_joints + len(keypoints_ext)

  root = 0

  center = 4

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    # extended
    'I3', 'M3', 'L3', 'R3', 'T3' #20
  ]


class SMPLArmature:
  n_joints = 24

  # indices of extended keypoints (limb ends)
  # lfinger, rfinger, ltoe, rtoe, head-top
  keypoints_ext = [2446, 5907, 3216, 6618, 411]

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'pelvis',
    'llegroot', 'rlegroot',
    'lowerback',
    'lknee', 'rknee',
    'upperback',
    'lankle', 'rankle',
    'thorax',
    'ltoes', 'rtoes',
    'lowerneck',
    'lclavicle', 'rclavicle',
    'upperneck',
    'larmroot', 'rarmroot',
    'lelbow', 'relbow',
    'lwrist', 'rwrist',
    'lhand', 'rhand'
    # extended
    'lfinger_tip', 'rfinger_tip', 'ltoe_tip', 'r_toe_tip', 'head_top'
  ]


class SMPLHArmature:
  n_joints = 52

  # indices of extended keypoints (limb ends)
  keypoints_ext = [
    2746, 2320, 2446, 2557, 2674,
    6191, 5781, 5907, 6018, 6135,
    3216, 6618, 411
  ]

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'pelvis',
    'llegroot', 'rlegroot',
    'lowerback',
    'lknee', 'rknee',
    'upperback',
    'lankle', 'rankle',
    'thorax',
    'ltoes', 'rtoes',
    'lowerneck',
    'lclavicle', 'rclavicle',
    'upperneck',
    'larmroot', 'rarmroot',
    'lelbow', 'relbow',
    'lwrist', 'rwrist',
    'lhand', 'rhand'
    # extended
    'left-thumb', 'li', 'lm', 'lr', 'll',
    'rt', 'ri', 'rm', 'rr', 'rl',
    'ltoe-tip', 'rtoe-tip', 'heat-top'
  ]

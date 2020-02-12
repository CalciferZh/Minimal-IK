# it must be assured that parent joint appears before child joint

class MANOArmature:
  n_joints = 16

  # indices of extended joints (finger tips)
  joints_ext = [333, 444, 672, 555, 744]

  n_keypoints = n_joints + len(joints_ext)

  root = 0

  center = 4

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]


class SMPLArmature:
  n_joints = 24

  # indices of extended joints (limb ends)
  # lfinger, rfinger, ltoe, rtoe, head-top
  joints_ext = [2446, 5907, 3216, 6618, 411]

  n_keypoints = n_joints + len(joints_ext)

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
  ]

# it must be assured that parent joint appears before child joint

class MANOArmature:
  # number of movable joints
  n_joints = 16

  # indices of extended keypoints
  keypoints_ext = [333, 444, 672, 555, 744]

  n_keypoints = n_joints + len(keypoints_ext)

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

  keypoints_ext = [2446, 5907, 3216, 6618, 411]

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'pelvis',
    'left leg root', 'right leg root',
    'lowerback',
    'left knee', 'right knee',
    'upperback',
    'left ankle', 'right ankle',
    'thorax',
    'left toes', 'right toes',
    'lowerneck',
    'left clavicle', 'right clavicle',
    'upperneck',
    'left armroot', 'right armroot',
    'left elbow', 'right elbow',
    'left wrist', 'right wrist',
    'left hand', 'right hand'
    # extended
    'left finger tip', 'right finger tip', 'left toe tip', 'right toe tip',
    'head_top'
  ]


class SMPLHArmature:
  n_joints = 52

  keypoints_ext = [
    2746, 2320, 2446, 2557, 2674,
    6191, 5781, 5907, 6018, 6135,
    3216, 6618, 411
  ]

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'pelvis',
    'left leg root', 'right leg root',
    'lowerback',
    'left knee', 'right knee',
    'upperback',
    'left ankle', 'right ankle',
    'thorax',
    'left toes', 'right toes',
    'lowerneck',
    'left clavicle', 'right clavicle',
    'upperneck',
    'left armroot', 'right armroot',
    'left elbow', 'right elbow',
    'left wrist', 'right wrist',
    'left hand', 'right hand'
    # extended
    'left thumb', 'left index', 'left middle', 'left ring', 'left little',
    'right thumb', 'right index', 'right middle', 'right ring', 'right little',
    'left toe tip', 'right toe tip', 'heat-top'
  ]

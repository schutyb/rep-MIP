import tifffile
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac

file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/pruebas algoritmos/alineacion/alignment_1.lsm')
im = tifffile.imread(file)
# img = phlib.concatenate(im, 2, 2, per=0)

# ########  Feature detection and matching #####################################################
# Initialize ORB
orb = ORB(n_keypoints=1000, fast_threshold=0.01)
# Detect keypoints in pano0
orb.detect_and_extract(im[0][:, 973:1024])
keypoints0 = orb.keypoints
descriptors0 = orb.descriptors
# Detect keypoints in pano1
orb.detect_and_extract(im[1][:, 0:51])
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors
matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)

'''fig, ax = plt.subplots(1, 1, figsize=(15, 12))
plot_matches(ax, im[0][:, 973:1024], im[1][:, 0:51], keypoints0, keypoints1, matches01)
ax.axis('off')
plt.show()
'''

# #######################3 Transform estimation ################################################
# Select keypoints from
#   * source (image to be registered): pano0
#   * target (reference image): pano1, our middle frame registration target
src = keypoints0[matches01[:, 0]][:, ::-1]
dst = keypoints1[matches01[:, 1]][:, ::-1]

model_robust01, inliers01 = ransac((src, dst), ProjectiveTransform,
                                   min_samples=3, residual_threshold=1, max_trials=1000)

''' fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_matches(ax, im[0][:, 973:1024], im[1][:, 0:51], keypoints0, keypoints1, matches01[inliers01])
ax.axis('off')
plt.show() '''


# ######################################### Warping ##############################################

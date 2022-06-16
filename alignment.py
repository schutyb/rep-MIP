import tifffile
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from skimage.measure import ransac
import numpy as np
import PhasorLibrary as ph


file = str('/home/bruno/Documentos/TESIS/TESIS/pruebas algoritmos/alineacion/alignment_1.lsm')
im = tifffile.imread(file)

im1 = im[0][:, 968:1019]
im2 = im[1][:, 0:51]

# ########  Feature detection and matching #####################################################
# Initialize ORB
orb = ORB(n_keypoints=1000, fast_threshold=0.01)
# Detect keypoints in pano0
orb.detect_and_extract(im1)
keypoints0 = orb.keypoints
descriptors0 = orb.descriptors
# Detect keypoints in pano1
orb.detect_and_extract(im2)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors
matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)

'''fig, ax = plt.subplots(1, 1, figsize=(15, 12))
plot_matches(ax, im1, im2, keypoints0, keypoints1, matches01)
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

'''fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_matches(ax, im1, im2, keypoints0, keypoints1, matches01[inliers01])
ax.axis('off')
plt.show()'''

# ######################################### Warping ##############################################
im_out = np.zeros(im1.shape)
r, c = im1.shape
corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])
warped_corners = model_robust01(corners)
all_corners = np.vstack((warped_corners, corners))
corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)
output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1]).astype(int)

# This in-plane offset is the only necessary transformation for the middle image
offset1 = SimilarityTransform(translation=-corner_min)

# Translate pano1 into place
im2_warped = warp(im2, offset1.inverse, order=3, output_shape=output_shape, cval=-1)

# Acquire the image mask for later use
im2_mask = (im2_warped != -1)  # Mask == 1 inside image
im2_warped[~im2_mask] = 0  # Return background values to 0

# Warp pano0 (left) to pano1
transform01 = (model_robust01 + offset1).inverse
im1_warped = warp(im1, transform01, order=3, output_shape=output_shape, cval=-1)

im1_mask = (im1_warped != -1)  # Mask == 1 inside image
im1_warped[~im1_mask] = 0  # Return background values to 0

# ALINEAMIENTO BRUTO  ###################################################################33
merged = (im1_warped + im2_warped)
overlap = (im1_mask * 1 + im2_mask)
normalized = merged / np.maximum(overlap, 1)

'''plt.figure(2)
plt.imshow(normalized, cmap='gray', interpolation='none')'''
# ########################################################################################

rmax = output_shape[0] - 1
cmax = output_shape[1] - 1

# Start anywhere along the top and bottom, left of center.
mask_pts01 = [[0, cmax // 3],
              [rmax, cmax // 3]]

from skimage.morphology import flood_fill


def generate_costs(diff_image, mask, vertical=True, gradient_cutoff=2.):
    """
    Ensures equal-cost paths from edges to region of interest.

    Parameters
    ----------
    diff_image : ndarray of floats
        Difference of two overlapping images.
    mask : ndarray of bools
        Mask representing the region of interest in ``diff_image``.
    vertical : bool
        Control operation orientation.
    gradient_cutoff : float
        Controls how far out of parallel lines can be to edges before
        correction is terminated. The default (2.) is good for most cases.

    Returns
    -------
    costs_arr : ndarray of floats
        Adjusted costs array, ready for use.
    """
    if vertical is not True:
        return tweak_costs(diff_image.T, mask.T, vertical=vertical,
                           gradient_cutoff=gradient_cutoff).T

    # Start with a high-cost array of 1's
    costs_arr = np.ones_like(diff_image)

    # Obtain extent of overlap
    row, col = mask.nonzero()
    cmin = col.min()
    cmax = col.max()
    shape = mask.shape

    # Label discrete regions
    labels = mask.copy().astype(np.uint8)
    cslice = slice(cmin, cmax + 1)
    submask = np.ascontiguousarray(labels[:, cslice])
    submask = flood_fill(submask, (0, 0), 2)
    submask = flood_fill(submask, (shape[0] - 1, 0), 3)
    labels[:, cslice] = submask

    # Find distance from edge to region
    upper = (labels == 2).sum(axis=0).astype(np.float64)
    lower = (labels == 3).sum(axis=0).astype(np.float64)

    # Reject areas of high change
    ugood = np.abs(np.gradient(upper[cslice])) < gradient_cutoff
    lgood = np.abs(np.gradient(lower[cslice])) < gradient_cutoff

    # Give areas slightly farther from edge a cost break
    costs_upper = np.ones_like(upper)
    costs_lower = np.ones_like(lower)
    costs_upper[cslice][ugood] = upper[cslice].min() / np.maximum(upper[cslice][ugood], 1)
    costs_lower[cslice][lgood] = lower[cslice].min() / np.maximum(lower[cslice][lgood], 1)

    # Expand from 1d back to 2d
    vdist = mask.shape[0]
    costs_upper = costs_upper[np.newaxis, :].repeat(vdist, axis=0)
    costs_lower = costs_lower[np.newaxis, :].repeat(vdist, axis=0)

    # Place these in output array
    costs_arr[:, cslice] = costs_upper[:, cslice] * (labels[:, cslice] == 2)
    costs_arr[:, cslice] += costs_lower[:, cslice] * (labels[:, cslice] == 3)

    # Finally, place the difference image
    costs_arr[mask] = diff_image[mask]

    return costs_arr


costs01 = generate_costs(np.abs(im1_warped - im2_warped),
                         im1_mask & im2_mask)

costs01[0, :] = 0
costs01[-1, :] = 0

from skimage.graph import route_through_array

pts, _ = route_through_array(costs01, mask_pts01[0], mask_pts01[1], fully_connected=True)
# Convert list of lists to 2d coordinate array for easier indexing
pts = np.array(pts)

'''plt.figure(3)
plt.imshow(im1_warped - im2_warped, cmap='gray')  # Plot the difference image
plt.plot(pts[:, 1], pts[:, 0])  # Overlay the minimum-cost path
plt.title('Minimum cost path')
plt.tight_layout()
plt.show()'''

mask1 = np.zeros_like(im1_warped, dtype=np.uint8)
mask1[pts[:, 0], pts[:, 1]] = 1

# con el codigo de abajo veo el camino y la imagen de la zona que agrego
'''plt.figure(4)
plt.imshow(mask0, cmap='gray')
plt.show()'''


from skimage.morphology import flood_fill
mask1 = flood_fill(mask1, (0, 0), 1, connectivity=1)

'''plt.figure(5)
plt.imshow(mask1, cmap='gray')
plt.show()'''


def add_alpha(img, mask=None):
    """
    Adds a masked alpha channel to an image.

    Parameters
    ----------
    img : (M, N[, 3]) ndarray
        Image data, should be rank-2 or rank-3 with RGB channels
    mask : (M, N[, 3]) ndarray, optional
        Mask to be applied. If None, the alpha channel is added
        with full opacity assumed (1) at all locations.
    """
    from skimage.color import gray2rgb
    if mask is None:
        mask = np.ones_like(img)

    if img.ndim == 2:
        img = gray2rgb(img)

    return np.dstack((img, mask))


mask2 = ~(mask1.astype(np.bool))
im1_final = add_alpha(im1_warped, mask1)
im2_final = add_alpha(im2_warped, mask2)

# grafico todx para comparar

fig, ax = plt.subplots(1, 5, figsize=(20, 12))
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(im2, cmap='gray')
ax[1].set_title('Imágenes originales')

ax[2].imshow(normalized, cmap='gray', interpolation='none')
ax[2].set_title('Primer método')

ax[3].imshow(im1_final, interpolation='none')
ax[3].imshow(im2_final, interpolation='none')
ax[3].set_title('Segundo método (Optimal path')

ax[4].imshow((im1+im2)/2, cmap='gray')
plt.show()

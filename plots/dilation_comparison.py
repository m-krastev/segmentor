from scipy.ndimage import binary_dilation
import torch
from navigator.utils import BinaryDilation3D, draw_path_sphere
import jax.numpy as jnp
from skimage.draw import line_nd
import matplotlib.pyplot as plt

img = torch.zeros((10, 10, 10), dtype=torch.float32)

line = line_nd((0, 0, 0), (10, 10, 10))
img[line] = 1

dilation_star = BinaryDilation3D("star")
dilation_ball = BinaryDilation3D("cube", kernel_size=3)

dilation_star = dilation_star.to(img.device)
dilation_ball = dilation_ball.to(img.device)

img_star = dilation_star(img.unsqueeze(0).unsqueeze_(0)).squeeze(0).squeeze(0)
img_ball = dilation_ball(img.unsqueeze(0).unsqueeze_(0)).squeeze(0).squeeze(0)

# JAX version
img_jax = jnp.from_dlpack(img, copy=False)
img_jax = draw_path_sphere(img_jax, line, radius=1, fill_value=1)[0]

# Scipy version
img_scipy = binary_dilation(img.cpu().numpy())

# Combined plots (2D)
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
axs[0].imshow(img[5].cpu().numpy(), cmap="Greens")
axs[0].set_title("Original")
axs[1].imshow(img_scipy[5], cmap="Oranges")
axs[1].set_title("Dilation")
axs[2].imshow(img_star[5].cpu().numpy(), cmap="Reds")
axs[2].set_title("Convolution (Star)")
axs[3].imshow(img_ball[5].cpu().numpy(), cmap="Blues")
axs[3].set_title("Convolution (Ball)")
axs[4].imshow(img_jax[5], cmap="Purples")
axs[4].set_title("Radial Expansion")
fig.savefig("dilation_comparison_2d.pdf", bbox_inches="tight")

# Combined plots (3D)
# Fix axes to be consistent
fig = plt.figure(figsize=(18, 6))  # Increased figure size for better spacing
ax1 = fig.add_subplot(151, projection="3d")
ax1.voxels(img.cpu().numpy(), facecolors="green", edgecolors="k")
ax1.set_title("Original")
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 10])
ax1.set_zlim([0, 10])

ax2 = fig.add_subplot(152, projection="3d")
ax2.voxels(img_scipy, facecolors="orange", edgecolors="k")
ax2.set_title("Dilation")
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 10])
ax2.set_zlim([0, 10])

ax3 = fig.add_subplot(153, projection="3d")
ax3.voxels(img_star.cpu().numpy(), facecolors="red", edgecolors="k")
ax3.set_title("Convolution (Star)")
ax3.set_xlim([0, 10])
ax3.set_ylim([0, 10])
ax3.set_zlim([0, 10])

ax4 = fig.add_subplot(154, projection="3d")
ax4.voxels(img_ball.cpu().numpy(), facecolors="blue", edgecolors="k")
ax4.set_title("Convolution (Cube)")
ax4.set_xlim([0, 10])
ax4.set_ylim([0, 10])
ax4.set_zlim([0, 10])


ax5 = fig.add_subplot(155, projection="3d")
ax5.voxels(img_jax, facecolors="purple", edgecolors="k")
ax5.set_title("Radial Expansion")
ax5.set_xlim([0, 10])
ax5.set_ylim([0, 10])
ax5.set_zlim([0, 10])
fig.savefig("dilation_comparison_3d.pdf", bbox_inches="tight")

# Now with repeated dilation
dilation_ball_repeated = torch.nn.Sequential(*[BinaryDilation3D("cube", 3)] * 3)
img_ball_repeated = dilation_ball_repeated(img.unsqueeze(0).unsqueeze_(0)).squeeze(0).squeeze(0)

img_jax_repeated = draw_path_sphere(img_jax, line, radius=3, fill_value=1)[0]
dilation_star_repeated = torch.nn.Sequential(*[BinaryDilation3D("star")] * 3)
img_star_repeated = dilation_star_repeated(img.unsqueeze(0).unsqueeze_(0)).squeeze(0).squeeze(0)
scipy_dilation_repeated = binary_dilation(img.cpu().numpy(), iterations=3)

# 2D
fig, axs = plt.subplots(1, 5, figsize=(12, 4))  # Increased figure size for better spacing
axs[0].imshow(img[5].cpu().numpy(), cmap="Greens")
axs[0].set_title("Original")
axs[1].imshow(scipy_dilation_repeated[5], cmap="Oranges")
axs[1].set_title("Dilation")
axs[2].imshow(img_star_repeated[5].cpu().numpy(), cmap="Reds")
axs[2].set_title("Convolution (Star)")
axs[3].imshow(img_ball_repeated[5].cpu().numpy(), cmap="Blues")
axs[3].set_title("Convolution (Ball)")
axs[4].imshow(img_jax_repeated[5], cmap="Purples")
axs[4].set_title("Radial Expansion")
fig.savefig("dilation_comparison_repeated_2d.pdf", bbox_inches="tight")
# 3D
fig = plt.figure(figsize=(18, 6))  # Increased figure size for better spacing
ax1 = fig.add_subplot(151, projection="3d")
ax1.voxels(img.cpu().numpy(), facecolors="green", edgecolors="k")
ax1.set_title("Original")
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 10])
ax1.set_zlim([0, 10])

ax2 = fig.add_subplot(152, projection="3d")
ax2.voxels(scipy_dilation_repeated, facecolors="orange", edgecolors="k")
ax2.set_title("Dilation")
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 10])
ax2.set_zlim([0, 10])
ax3 = fig.add_subplot(153, projection="3d")
ax3.voxels(img_star_repeated.cpu().numpy(), facecolors="red", edgecolors="k")
ax3.set_title("Convolution (Star)")
ax3.set_xlim([0, 10])
ax3.set_ylim([0, 10])
ax3.set_zlim([0, 10])
ax4 = fig.add_subplot(154, projection="3d")
ax4.voxels(img_ball_repeated.cpu().numpy(), facecolors="blue", edgecolors="k")
ax4.set_title("Convolution (Ball)")
ax4.set_xlim([0, 10])
ax4.set_ylim([0, 10])
ax4.set_zlim([0, 10])
ax5 = fig.add_subplot(155, projection="3d")
ax5.voxels(img_jax_repeated, facecolors="purple", edgecolors="k")
ax5.set_title("Radial Expansion")
ax5.set_xlim([0, 10])
ax5.set_ylim([0, 10])
ax5.set_zlim([0, 10])
fig.savefig("dilation_comparison_repeated_3d.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()

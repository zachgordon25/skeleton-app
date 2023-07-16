import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image  # ZG
from skeleton.BlumMedialAxis import BlumMedialAxis
from scipy.spatial import Delaunay

# Assuming you have the BlumMedialAxis class defined

# pathin = "M-to-PY_Skel/original"
# pathout = "M-to-PY_Skel/skeleton"

# pics = [f for f in os.listdir(pathin) if f.endswith(".txt")]
# for _, pic in enumerate(pics):
#     print(_, pic)
#     rough = np.loadtxt(os.path.join(pathin, pic))
#     boundary = rough[:, 0] + 1j * rough[:, 1]
#     bma = BlumMedialAxis(boundary)

#     # Assuming you have a method called plot_with_edges() in BlumMedialAxis class
#     bma.plot_with_edges()
#     # bma.plot_with_wedf()

#     filename = pic.replace("coords_", "").replace(".txt", "")
#     plt.title(filename)

#     plt.savefig(os.path.join(pathout, f"SkeletonWithBoundary_{pic}.png"))  # ZG

#     img = Image.open(os.path.join(pathout, f"SkeletonWithBoundary_{pic}.png"))  # ZG
#     img.save(os.path.join(pathout, f"SkeletonWithBoundary_{pic}.gif"), "gif")  # ZG
#     plt.close()

#     os.remove(os.path.join(pathout, f"SkeletonWithBoundary_{pic}.png"))  # ZG


# print("#2 runs")


def generate_skeleton(contour_strings, filename):
    # Convert contour_strings into numpy array
    rough = np.array([list(map(float, s.split())) for s in contour_strings])

    boundary = rough[:, 0] + 1j * rough[:, 1]
    bma = BlumMedialAxis(boundary)

    # Assuming you have a method called plot_with_edges() in BlumMedialAxis class
    bma.plot_with_edges()

    plt.title(filename)

    # Generate a unique file name
    output_file = os.path.join("skeleton", f"SkeletonWithBoundary_{filename}.png")

    plt.savefig(output_file)

    img = Image.open(output_file)
    output_gif = output_file.replace(".png", ".gif")
    img.save(output_gif, "gif")
    plt.close()

    # Remove the png file
    os.remove(output_file)

    return output_gif

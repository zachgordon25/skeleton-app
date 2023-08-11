import base64
from io import BytesIO

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skeleton.BlumMedialAxis import BlumMedialAxis


def generate_skeleton(contour_strings, filename):
    # Convert contour_strings into numpy array
    rough = np.array([list(map(float, s.split())) for s in contour_strings])

    boundary = rough[:, 0] + 1j * rough[:, 1]
    bma = BlumMedialAxis(boundary)

    # Assuming you have a method called plot_with_edges() in BlumMedialAxis class
    bma.plot_with_edges()

    plt.title(filename)

    # Create buffer
    buf = BytesIO()

    # Save figure to buffer in PNG format
    plt.savefig(buf, format="png")

    # Save the figure to the history directory
    plt.savefig(f"static/history/{filename}.png")

    # Get buffer contents and encode in base64
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()

    return image_base64

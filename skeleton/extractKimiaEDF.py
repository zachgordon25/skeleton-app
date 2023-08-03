# Standard library imports
import base64
from io import BytesIO

# Related third-party imports
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local application/library specific imports
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

    # Get buffer contents and encode in base64
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()

    return image_base64

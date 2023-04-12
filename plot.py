import numpy as np
from nptyping import NDArray, Float64, Shape
import matplotlib.pyplot as plt

class Plot(object):

    @staticmethod
    def matrix(data: NDArray[Shape["*, *"], Float64]) -> None:
        fig, ax = plt.subplots()
        plt.imshow(data)
        plt.colorbar()
        plt.show()

    @staticmethod
    def normalized_mono_image(data: NDArray[Shape["*, *"], Float64]) -> None:
        data = data / np.amax(data)
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.imshow(data, cmap="gray", vmin=0, vmax=1)
        plt.show()

    @staticmethod
    def normalized_image_histogram(data: NDArray[Shape["*, *"], Float64]) -> None:
        data = data / np.amax(data)
        histogram, bin_edges = np.histogram(data, bins=256, range=(0, 1))
        plt.figure()
        plt.title("Grayscale histogram")
        plt.xlabel("Grayscale value")
        plt.ylabel("Pixel count")
        plt.xlim([0.0, 1.0])
        plt.plot(bin_edges[0:-1], histogram)
        plt.show()

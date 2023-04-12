import numpy as np
from scipy.special import binom
from nptyping import NDArray, Float64, Shape

class Zernike_generator():

    def __init__(self, matrix_size: int, xy_space_extent: float, fractional_pupil_radius: float = 1):
        """
        Creates an instance of the Zernike aberration generator.

        Parameters
        ----------
        matrix_size : int
            Size of a side of the aberration matrix in pixels. The matrix is always a square.

        xy_space_extent : float
            Determines the scale of the aberrations and the PSF.
        """

        self._matrix = np.zeros((matrix_size, matrix_size), dtype=float)
        self._xy_space_extent = xy_space_extent
        x = y = np.linspace(-self._xy_space_extent, self._xy_space_extent, matrix_size)
        self._mesh_X, self._mesh_Y = np.meshgrid(x, y)
        self._rho = np.sqrt(self._mesh_X ** 2 + self._mesh_Y ** 2)
        self._theta = np.arctan2(self._mesh_Y, self._mesh_X)
        self._fractional_pupil_radius = fractional_pupil_radius
        self._mask = np.where(self._rho > self._xy_space_extent * self._fractional_pupil_radius, 0, 1)

    def _nm_normalization(self, n: int, m: int):
        """
        The normalization of the Zernike mode (n, m) using the Born/Wolf convetion.

        Example: sqrt( \int | z_nm |^2 )
        """
        return np.sqrt((1.+(m == 0))/(2.*n+2))

    def _nm_polynomial(self, n: int, m: int, normalized = True):
        """
        Returns the Zernike polyonimal by classical (n, m) enumeration.

        if normalized = True, then they form an orthonormal system
                \int z_nm z_n'm' = delta_nn' delta_mm'
                and the first modes are
                z_nm(0,0)  = 1/sqrt(pi)*
                z_nm(1,-1) = 1/sqrt(pi)* 2r cos(phi)
                z_nm(1,1)  = 1/sqrt(pi)* 2r sin(phi)
                z_nm(2,0)  = 1/sqrt(pi)* sqrt(3)(2 r^2 - 1)
                ...
                z_nm(4,0)  = 1/sqrt(pi)* sqrt(5)(6 r^4 - 6 r^2 +1)
                ...
        if normalized = False, then they follow the Born/Wolf convention
                (i.e. min/max is always -1/1)
                \int z_nm z_n'm' = (1.+(m==0))/(2*n+2) delta_nn' delta_mm'
                z_nm(0,0)  = 1
                z_nm(1,-1) = r cos(phi)
                z_nm(1,1)  =  r sin(phi)
                z_nm(2,0)  = (2 r^2 - 1)
                ...
                z_nm(4,0)  = (6 r^4 - 6 r^2 +1)


        This method is based on Andrea Bassi’s code:
        https://github.com/andreabassi78/napari-psf-simulator
        """

        if abs(m) > n:
            raise ValueError(" |m| <= n ! ( %s <= %s)" % (m, n))

        if (n - m) % 2 == 1:
            return 0 * self._rho + 0 * self._theta

        radial = 0
        m0 = abs(m)

        for k in range((n - m0) // 2 + 1):
            radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 *
                        k, (n - m0) // 2 - k) * self._rho ** (n - 2 * k)

        radial = radial * (self._rho <= self._xy_space_extent * self._fractional_pupil_radius)

        if normalized:
            prefac = 1. / self._nm_normalization(n, m)
        else:
            prefac = 0.5
        if m >= 0:
            return prefac * radial * np.cos(m0 * self._theta)
        else:
            return prefac * radial * np.sin(m0 * self._theta)

    def add_aberration(self, N: int, M: int, weight: float) -> None:
        """
        Adds the specified (N, M) aberration to the matrix.

        Parameters
        ----------
        N : int
            The N index of the (N, M) Born/Wolf convetion.

        M : int
            The M index of the (N, M) Born/Wolf convetion.

        weight : float
            The weight of the polynomials in the units of lambda. Weight of 1 means the wavefront is abberated with lambda / 2.

        """

        self._matrix += 2 * np.pi * weight * self._nm_polynomial(N, M, normalized = False)

    def get_raw_matrix(self) -> NDArray[Shape["*, *"], Float64]:
        """
		Returns
		-------
    	NDArray[Shape["*, *"], Float64]
        	A two-dimensional NDArray of Float64 values representing the aberration matrix.
        """
        return self._matrix

    def get_psf(self) -> NDArray[Shape["*, *"], Float64]:
        """
		Returns
		-------
    	NDArray[Shape["*, *"], Float64]
        	A two-dimensional array of Float64 values representing the point spread function created from the aberration matrix.
        """
        psf = np.exp(1j * self._matrix) * self._mask
        return np.abs(np.fft.fftshift(np.fft.fft2(psf))) ** 2

    def get_normalized_psf(self) -> NDArray[Shape["*, *"], Float64]:
        """
		Returns
		-------
    	NDArray[Shape["*, *"], Float64]
        	A two-dimensional array of Float64 values representing the normalized point spread function.
        """
        psf = self.get_psf()
        return psf / np.amax(psf)

    def _insert_to_array_at(self, main_array: NDArray[Shape["*, *"], Float64], array_to_be_inserted: NDArray[Shape["*, *"], Float64], x: int, y: int) -> NDArray[Shape["*, *"], Float64]:
        """
        Inserts 'array_to_be_inserted' into 'main_array'.

        Parameters
        ----------
        x : int
            The X coordinate.

        y : int
            The Y coordinate.

        main_array : NDArray[Shape["*, *"], Float64]
            The main array.

        array_to_be_inserted : NDArray[Shape["*, *"], Float64]
            The array to be inserted.

        Returns
        -------
        NDArray[Shape["*, *"], Float64]
            The resulting array.
        """

        if main_array.ndim != 2 or array_to_be_inserted.ndim != 2:
            raise Exception("Invalid array size.")

        x2 = x + array_to_be_inserted.shape[0]
        y2 = y + array_to_be_inserted.shape[1]

        assert x2 <= main_array.shape[0], "The position will make the smaller matrix exceed the boundaries at x."
        assert y2 <= main_array.shape[1], "The position will make the smaller matrix exceed the boundaries at y."

        main_array[x:x2, y:y2] = array_to_be_inserted

        return main_array

    def apply_psf_to_image(self, image_data: NDArray[Shape["*, *"], Float64]) -> NDArray[Shape["*, *"], Float64]:
        """
        Apply the point spread function to the given image. 

        Parameters
        ----------
        image_data : NDArray[Shape["*, *"], Float64]
            A normalized single-channel image.

        Returns
        -------
        NDArray[Shape["*, *"], Float64]
            The resulting image.
        """
        if image_data.ndim != 2 or image_data.shape[0] < self._matrix.shape[0] or image_data.shape[1] < self._matrix.shape[1]:
            raise Exception("Invalid image array size.")

        psf = self.get_normalized_psf()
        psf_with_padding = np.zeros((image_data.shape[0], image_data.shape[1]), dtype=float)

        left = int((image_data.shape[0] - psf.shape[0]) / 2)
        top = int((image_data.shape[1] - psf.shape[1]) / 2)

        psf_with_padding = self._insert_to_array_at(psf_with_padding, psf, left, top)

        # Convert the image into a Fourier domain. 
        f_image = np.fft.fftshift(np.fft.fft2(image_data))

        # Optical transform function
        otf = np.fft.fftshift(np.fft.fft2(psf_with_padding))

        # Element-wise multiplication
        spectra = f_image * otf

        final_image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(spectra))))

        return final_image / final_image.max()

if __name__ == '__main__':

    # Create an instance of the generator
    generator = Zernike_generator(64, 1)

    # Apply defocus
    generator.add_aberration(2, 0, 1)

    # Apply vertical astigmatism
    generator.add_aberration(2, 2, 1)

    # Apply horizontal tilt
    generator.add_aberration(1, 1, 1)

    # Apply oblique astigmatism
    generator.add_aberration(2, -2, 1)

    # Plot the point spread function
    import matplotlib.pyplot as plt
    plt.imshow(generator.get_normalized_psf())
    plt.colorbar()
    plt.show()

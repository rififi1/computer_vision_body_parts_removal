import cv2
import numpy as np
import matplotlib.pyplot as plt


class SkinMask:
    def __init__(self, debug=False):
        self.debug = debug
        pass

    def YCrCb_img(self, img):
        """
        Convert an image from RGB to YCrCb color space and rescale the channels to match the expected ranges for further processing.

        Parameters
        ----------
        img : numpy.ndarray
                                        Input image in RGB format.
        Returns
        -------
        y : numpy.ndarray
                                        Y channel in float32 format, rescaled to the range [16, 235].
        cr : numpy.ndarray
                                        Cr channel in float32 format, rescaled to the range [16, 240].
        cb : numpy.ndarray
                                        Cb channel in float32 format, rescaled to the range [16, 240].
        """

        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ## Extract the Y, Cr and Cb channels
        y, cr, cb = cv2.split(img_ycrcb)
        y, cr, cb = y.astype(np.float32), cr.astype(np.float32), cb.astype(np.float32)

        ## Debug: Print ranges of input channels
        if self.debug:
            print(
                f"Input ranges - Y: [{y.min():.1f}, {y.max():.1f}], Cr: [{cr.min():.1f}, {cr.max():.1f}], Cb: [{cb.min():.1f}, {cb.max():.1f}]"
            )

        ## Adjust Y: full range [0,255] -> video range [16,235]
        y = y * (219.0 / 255.0) + 16.0

        ## Adjust Cr and Cb: full range [0,255] -> video range [16,240]
        cr = cr * (224.0 / 255.0) + 16.0
        cb = cb * (224.0 / 255.0) + 16.0

        ## Debug: Print ranges after video range adjustment
        if self.debug:
            print(
                f"After video range - Y: [{y.min():.1f}, {y.max():.1f}], Cr: [{cr.min():.1f}, {cr.max():.1f}], Cb: [{cb.min():.1f}, {cb.max():.1f}]"
            )
        return y, cr, cb

    def HSV_img(self, img):
        """
        Convert an image from RGB to HSV color space and rescale the channels to match the expected ranges for further processing.

        Parameters
        ----------
        img : numpy.ndarray
                                        Input image in RGB format.

        Returns
        -------
        H_deg : numpy.ndarray
                                        Hue channel in degrees [0..358].
        S_pct : numpy.ndarray
                                        Saturation channel in percentage [0..100].
        V_pct : numpy.ndarray
                                        Value channel in percentage [0..100].
        """

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        ## Split H,S,V as 8-bit unsigned
        H_8u, S_8u, V_8u = cv2.split(img_hsv)

        ## Convert to float32 for calculations
        H = H_8u.astype(np.float32)
        S = S_8u.astype(np.float32)
        V = V_8u.astype(np.float32)

        if self.debug:
            print(
                f"Input ranges - H: [{H.min():.1f}, {H.max():.1f}], S: [{S.min():.1f}, {S.max():.1f}], V: [{V.min():.1f}, {V.max():.1f}]"
            )

        ## Rescale:
        ##   OpenCV's H in [0..179] -> degrees in [0..358] (approx)
        ##   OpenCV's S,V in [0..255] -> percentages in [0..100]
        H_deg = 2.0 * H
        S_pct = (S / 255.0) * 100
        V_pct = (V / 255.0) * 100

        if self.debug:
            print(
                f"After rescaling - H: [{H.min():.1f}, {H.max():.1f}], S: [{S.min():.1f}, {S.max():.1f}], V: [{V.min():.1f}, {V.max():.1f}]"
            )

        return H_deg, S_pct, V_pct

    def naive_otsu(self, img):
        """
        Creates a skin mask in the image using the naive Otsu thresholding method.

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        _, cr, _ = cv2.split(ycrcb)
        cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin_mask = cv2.threshold(
            cr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return skin_mask

    def YCrCb_garcia(self, img):
        """
        Creates a skin mask in the image using the YCrCb color space thresholding
        following the method described in the paper 'Face Detection Using Quantized
        Skin Color Regions Merging and Wavelet Packet Analysis' (Garcia and Tziritas, 1999).

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """
        y, cr, cb = self.YCrCb_img(img)
        ## define the theta variables
        theta1, theta2, theta3, theta4 = (np.zeros_like(y, dtype=np.float32),) * 4

        ## define mask high and low thresholds
        mask_high = y > 128
        mask_low = ~mask_high

        ## Set thetas for mask_high
        theta1[mask_high] = -2.0 + (256.0 - y[mask_high]) / 16.0
        theta2[mask_high] = 20.0 - (256.0 - y[mask_high]) / 16.0
        theta3[mask_high] = 6.0
        theta4[mask_high] = -8.0

        ## Set thetas for mask_low
        theta1[mask_low] = 6.0
        theta2[mask_low] = 12.0
        theta3[mask_low] = 2.0 + y[mask_low] / 32.0
        theta4[mask_low] = -16.0 + y[mask_low] / 16.0

        ## Debug: Print ranges of theta values
        if self.debug:
            print(
                f"Theta ranges - theta1: [{theta1.min():.1f}, {theta1.max():.1f}], theta2: [{theta2.min():.1f}, {theta2.max():.1f}], theta3: [{theta3.min():.1f}, {theta3.max():.1f}], theta4: [{theta4.min():.1f}, {theta4.max():.1f}]"
            )

        ## Set constraints from the paper
        # Basic constraints
        cond1 = cr >= -2.0 * (cb + 24.0)
        cond2 = cr >= -(cb + 17.0)
        cond3 = cr >= -4.0 * (cb + 32.0)
        cond4 = cr >= 2.5 * (cb + theta1)
        cond5 = cr >= theta3
        cond6 = cr >= 0.5 * (4.0 - cb)
        cond7 = cr <= 220.0 - (cb / 6.0)
        cond8 = cr <= (4.0 / 3.0) * (2.0 - cb)

        ## Debug: Print percentage of pixels satisfying each condition
        if self.debug:
            print("\nPercentage of pixels satisfying each condition:")
            print(f"cond1: {np.mean(cond1)*100:.1f}%")
            print(f"cond2: {np.mean(cond2)*100:.1f}%")
            print(f"cond3: {np.mean(cond3)*100:.1f}%")
            print(f"cond4: {np.mean(cond4)*100:.1f}%")
            print(f"cond5: {np.mean(cond5)*100:.1f}%")
            print(f"cond6: {np.mean(cond6)*100:.1f}%")
            print(f"cond7: {np.mean(cond7)*100:.1f}%")
            print(f"cond8: {np.mean(cond8)*100:.1f}%")

        ## combine all constraints
        skin_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8

        ## Debug: Print final mask statistics
        if self.debug:
            print(
                f"\nFinal mask - pixels detected: {np.sum(skin_mask)} ({np.mean(skin_mask)*100:.1f}%)"
            )

        ## Convert to uint8
        skin_mask_uint8 = skin_mask.astype(np.uint8) * 255

        return skin_mask_uint8

    def HSV_garcia(self, img):
        """
        Creates a skin mask in the image using the HSV color space thresholding
        following the method described in the paper 'Face Detection Using Quantized
        Skin Color Regions Merging and Wavelet Packet Analysis' (Garcia and Tziritas, 1999).

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """

        # Convert from RGB to HSV (if your img is BGR, use COLOR_BGR2HSV instead)
        H_deg, S_pct, V_pct = self.HSV_img(img)

        # Build constraints
        cond1 = S_pct >= 10
        cond2 = V_pct >= 40
        cond3 = S_pct <= (-H_deg - 0.1 * V_pct + 110)
        cond4 = H_deg <= -0.4 * V_pct + 75

        ## mask based on sign of H_deg
        mask_posH = H_deg >= 0
        mask_negH = ~mask_posH

        # Cond5 for H >= 0
        cond5a = S_pct <= (0.08 * (100.0 - V_pct) * H_deg + 0.5 * V_pct)
        ## Cond5 for H < 0
        cond5b = S_pct <= (0.5 * H_deg + 35.0)

        ## Combine them into a single cond5
        cond5 = np.zeros_like(cond1, dtype=bool)
        cond5[mask_posH] = cond5a[mask_posH]
        cond5[mask_negH] = cond5b[mask_negH]

        ## Final mask
        skin_mask = cond1 & cond2 & cond3 & cond4 & cond5

        skin_mask_uint8 = skin_mask.astype(np.uint8) * 255

        # Apply morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)
        skin_mask_uint8 = cv2.morphologyEx(skin_mask_uint8, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((7, 7), np.uint8)
        skin_mask_uint8 = cv2.morphologyEx(skin_mask_uint8, cv2.MORPH_CLOSE, kernel)

        return skin_mask_uint8

    def density_map_chai(self, color_mask):
        """
        Compute density map D(x,y) as described in the paper.
        It partitions the skin mask into 4x4 non-overlapping groups and counts skin pixels in each group.

        Parameters:
        - color_mask (numpy.ndarray): Binary mask where 255 indicates skin pixels, 0 non-skin

        Returns:
        - density_type (numpy.ndarray): Classification of each point (0: zero, 1: full)
        """
        ## get dimensions
        M, N = color_mask.shape

        # Calculate density map dimensions (M/4 × N/4 as per paper)
        M_d = M // 4
        N_d = N // 4

        # Initialize density map
        density_map = np.zeros((M_d, N_d), dtype=np.uint8)

        # For each point in density map
        for x in range(M_d):
            for y in range(N_d):
                # Initialize count for this 4x4 group
                count = 0

                # Sum over 4x4 group (i,j from 0 to 3 as per equation 2)
                for i in range(4):
                    for j in range(4):
                        # Get corresponding pixel in original mask
                        # Note: we multiply x,y by 4 to get to the start of each group
                        pixel_x = 4 * x + i
                        pixel_y = 4 * y + j

                        # Count if it's a skin pixel (255 in the mask)
                        if color_mask[pixel_x, pixel_y] == 255:
                            count += 1

                # Assign count to density map
                density_map[x, y] = count

        # Create density type map (0: zero, 1: intermediate, 2: full)
        bitmap = density_map == 16
        return bitmap

    def luminance_regularization_chai(self, luminance, density_mask):
        """
        Compute standard deviation of luminance values for 4x4 pixel regions in the image.

        Parameters:
        - luminance (numpy.ndarray): Luminance (Y) channel from YCrCb color space
        - density_mask (numpy.ndarray): Binary mask from stage 2 (density map)

        Returns:
        - luminance_mask (numpy.ndarray): Final binary mask after applying equation (5)
        """
        ## Get dimensions
        M, N = luminance.shape
        M_d = M // 4
        N_d = N // 4

        ## Initialize std dev map
        std_dev_map = np.zeros((M_d, N_d), dtype=np.float32)

        # For each point in the density map
        for x in range(M_d):
            for y in range(N_d):
                ## Get 4x4 window of luminance values
                window = luminance[4 * x : 4 * (x + 1), 4 * y : 4 * (y + 1)]

                # Calculate E[W] and E[W²]
                E_W = np.mean(window)
                E_W2 = np.mean(window**2)

                ## Calculate standard deviation using equation (4)
                std_dev_map[x, y] = np.sqrt(E_W2 - E_W**2)

        luminance_mask = np.zeros_like(density_mask, dtype=np.uint8)
        luminance_mask = (density_mask == 1) & (std_dev_map >= 0.5)

        return luminance_mask

    def apply_neighborhood_rules(self, mask):
        """
        Apply 3x3 neighborhood rules to remove noise:
        - Keep 1s if they have more than 3 neighbors with value 1
        - Convert 0s to 1s if they have more than 5 neighbors with value 1

        Parameters:
        - mask (numpy.ndarray): Binary mask from stage 3

        Returns:
        - filtered_mask (numpy.ndarray): Mask after applying neighborhood rules
        """
        # Create output mask
        filtered_mask = mask.copy()

        # Get mask dimensions
        rows, cols = mask.shape

        # Process each pixel (excluding borders)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Get 3x3 neighborhood
                neighborhood = mask[i - 1 : i + 2, j - 1 : j + 2]
                neighbor_sum = np.sum(neighborhood) - mask[i, j]  # Sum excluding center

                if mask[i, j] == 1:
                    # Keep 1 only if it has more than 3 neighbors with value 1
                    if neighbor_sum <= 2:
                        filtered_mask[i, j] = 0
                else:
                    # Convert 0 to 1 if it has more than 5 neighbors with value 1
                    if neighbor_sum > 3:
                        filtered_mask[i, j] = 1

        return filtered_mask

    def remove_short_runs(self, mask, min_run_length=4):
        """
        Remove short runs of 1s in both horizontal and vertical directions.

        Parameters:
        - mask (numpy.ndarray): Binary mask
        - min_run_length (int): Minimum length of continuous run to keep

        Returns:
        - cleaned_mask (numpy.ndarray): Mask after removing short runs
        """
        cleaned_mask = mask.copy()
        rows, cols = mask.shape

        # Horizontal scanning
        for i in range(rows):
            run_start = -1
            run_length = 0

            for j in range(cols):
                if mask[i, j] == 1:
                    if run_start == -1:
                        run_start = j
                    run_length += 1
                elif run_start != -1:
                    # End of run found
                    if run_length < min_run_length:
                        cleaned_mask[i, run_start:j] = 0
                    run_start = -1
                    run_length = 0

            # Check last run in row
            if run_start != -1 and run_length < min_run_length:
                cleaned_mask[i, run_start:] = 0

        # Vertical scanning
        for j in range(cols):
            run_start = -1
            run_length = 0

            for i in range(rows):
                if cleaned_mask[i, j] == 1:
                    if run_start == -1:
                        run_start = i
                    run_length += 1
                elif run_start != -1:
                    # End of run found
                    if run_length < min_run_length:
                        cleaned_mask[run_start:i, j] = 0
                    run_start = -1
                    run_length = 0

            # Check last run in column
            if run_start != -1 and run_length < min_run_length:
                cleaned_mask[run_start:, j] = 0

        return cleaned_mask

    def geometric_correction(self, luminance_mask):
        """
        Apply geometric correction (stage 4) to remove noise and ensure correct facial region shape.
        Process includes:
        1. Apply 3x3 neighborhood rules
        2. Remove short runs in horizontal and vertical directions

        Parameters:
        - luminance_mask (numpy.ndarray): Binary mask from stage 3

        Returns:
        - geometric_corrected_mask (numpy.ndarray): Final corrected mask
        """
        ## Apply neighborhood rules
        filtered_mask = self.apply_neighborhood_rules(luminance_mask)

        ## Remove short runs (threshold of 4 pixels for CIF-size image as mentioned in paper)
        geometric_corrected_mask = self.remove_short_runs(
            filtered_mask, min_run_length=2
        )

        return geometric_corrected_mask

    def upscale_with_edge_info(self, geometric_mask, color_mask):
        """
        Upscale the geometric mask back to original resolution using edge information from color segmentation.

        Parameters:
        - geometric_mask (numpy.ndarray): Binary mask from geometric correction (stage 4)
        - color_mask (numpy.ndarray): Original color segmentation mask from stage 1

        Returns:
        - final_mask (numpy.ndarray): Final upscaled mask
        """
        # Get dimensions of original image
        M, N = color_mask.shape

        # Initialize final mask
        final_mask = np.zeros((M, N), dtype=np.uint8)

        # For each point in geometric mask
        for x in range(geometric_mask.shape[0]):
            for y in range(geometric_mask.shape[1]):
                if geometric_mask[x, y] == 1:
                    # Map to corresponding 4x4 region in original resolution
                    x_start, x_end = 4 * x, 4 * (x + 1)
                    y_start, y_end = 4 * y, 4 * (y + 1)

                    # Copy values from original color segmentation for this region
                    final_mask[x_start:x_end, y_start:y_end] = color_mask[
                        x_start:x_end, y_start:y_end
                    ]

        return final_mask

    def ycrcb_chai(self, img, cr_thresholds=(133, 178), cb_thresholds=(77, 127)):
        ## get YCrCb segmentation (we disregard Y)
        y, cr, cb = self.YCrCb_img(img)

        ## Step 1 : color segmentation
        cr_min, cr_max = cr_thresholds
        cb_min, cb_max = 70, 150  # Expanded range for cooler tones through plastic
        color_mask = (
            (cr >= cr_min)
            & (cr <= cr_max)
            & (cb >= cb_min)
            & (cb <= cb_max)
            & (y <= 230)
        )
        color_mask = color_mask.astype(np.uint8) * 255

        ## Step 2 : density regularization
        density_mask = self.density_map_chai(color_mask)

        ## Step 3 : luminance regularization
        luminance_mask = self.luminance_regularization_chai(y, density_mask)

        ## Step 4: geometric correction
        geometric_corrected_mask = self.geometric_correction(luminance_mask)

        ## Step 5: Upscale using edge information from color segmentation
        final_mask = self.upscale_with_edge_info(geometric_corrected_mask, color_mask)

        return final_mask.astype(np.uint8) * 255

    def compute(self, img, method="naive_otsu"):
        methods = {
            "naive_otsu": self.naive_otsu,
            "ycrcb_garcia": self.YCrCb_garcia,
            "hsv_garcia": self.HSV_garcia,
            "ycrcb_chai": self.ycrcb_chai,
        }
        try:
            return methods[method](img)
        except KeyError:
            raise ValueError(f"Invalid method for skin mask: {method}")

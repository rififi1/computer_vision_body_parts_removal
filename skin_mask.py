import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

    def HSV_img(self, quantized_img):
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

        ## Split H,S,V as 8-bit unsigned
        H_8u, S_8u, V_8u = cv2.split(quantized_img)

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

    def quantize(self, img, n_clusters=16, space="hsv"):
        """
        Quantize the HSV image into 16 clusters.
        """

        # Convert image to HSV
        if space == "hsv":
            img_space = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif space == "ycrcb":
            img_space = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # Flatten the image
        img_space_flat = img_space.reshape(-1, 3)

        # Quantize the image
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(img_space_flat)

        # Get the cluster centers
        centers = kmeans.cluster_centers_

        # Reshape the centers to the original image shape
        quantized = centers[kmeans.labels_].reshape(img.shape).astype(np.uint8)
        if self.debug:
            plt.figure(figsize=(15, 5))
            print(quantized.shape)
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.subplot(1, 2, 2)
            if space == "hsv":
                plt.imshow(cv2.cvtColor(quantized, cv2.COLOR_HSV2RGB))
            elif space == "ycrcb":
                plt.imshow(cv2.cvtColor(quantized, cv2.COLOR_YCrCb2RGB))
            plt.title("Quantized Image")
            plt.show()
        quantized = quantized

        return quantized

    def naive_otsu(self, img, quantized=False):
        """
        Creates a skin mask in the image using the naive Otsu thresholding method.

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """
        if quantized:
            channels = self.quantize(img, space="ycrcb")
        else:
            channels = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        _, cr, _ = cv2.split(channels)
        if not quantized:
            cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
        else:
            cr_blur = cr
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

    def HSV_filter_garcia(self, quantized_img):
        """
        Creates a skin mask in the image using the HSV color space thresholding
        following the method described in the paper 'Face Detection Using Quantized
        Skin Color Regions Merging and Wavelet Packet Analysis' (Garcia and Tziritas, 1999).

        Parameters:
        - quantized_img (numpy.ndarray): Input image in quantized HSV format.
        """
        # Convert from RGB to HSV (if your img is BGR, use COLOR_BGR2HSV instead)
        H_deg, S_pct, V_pct = self.HSV_img(quantized_img)

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
        return skin_mask_uint8

    def merge_adjacent_color_regions(self, quantized_img, skin_mask):
        """
        Merge adjacent regions of the same color in the quantized image.

        Parameters:
        - quantized_img (numpy.ndarray): Quantized HSV image
        - skin_mask (numpy.ndarray): Binary mask where 255 indicates skin pixels

        Returns:
        - merged_regions (dict): Dictionary mapping color tuples to their merged region statistics
        """
        # Create binary mask of skin pixels
        skin_binary = skin_mask > 127

        # Apply the skin mask to the quantized image
        masked_quantized = quantized_img.copy()
        masked_quantized[~skin_binary] = [0, 0, 0]

        # Get unique colors (excluding background)
        unique_colors = np.unique(masked_quantized.reshape(-1, 3), axis=0)
        unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]

        if self.debug:
            print(f"Found {len(unique_colors)} unique skin colors")

        # Initialize results dictionary
        merged_regions = {}

        # Process each unique color to get its merged regions
        for color in unique_colors:
            color_tuple = tuple(color)

            # Create binary mask for this color
            color_mask = np.all(
                masked_quantized == color.reshape(1, 1, 3), axis=2
            ).astype(np.uint8)

            # Apply morphological closing to merge close regions of the same color
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # Now get connected components for the regions
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                color_mask
            )

            # Store stats for each component with this color
            merged_color_stats = []
            for i in range(1, num_labels):  # Skip background label 0
                x, y, w, h, area = stats[i]
                merged_color_stats.append(
                    {
                        "label": i,
                        "bbox": (x, y, w, h),
                        "area": area,
                        "centroid": centroids[i],
                    }
                )

            # Store all merged instances of this color
            merged_regions[color_tuple] = merged_color_stats

        if self.debug:
            total_regions = sum(len(regions) for regions in merged_regions.values())
            print(f"{total_regions} total regions")

            # Create visualization
            vis_img = cv2.cvtColor(quantized_img, cv2.COLOR_HSV2RGB).copy()

            # Draw bounding boxes for merged regions
            for color, regions in merged_regions.items():
                # Convert HSV color to RGB for visualization
                hsv_color = np.array([[color]], dtype=np.uint8)
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
                box_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

                for region in regions:
                    x, y, w, h = region["bbox"]
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), box_color, 2)

            plt.figure(figsize=(12, 8))
            plt.imshow(vis_img)
            plt.title("Homogenous Color Regions")
            plt.axis("off")
            plt.show()

        return merged_regions

    def color_similarity(self, color1, color2, alpha=5.0, beta=1.0, gamma=1.0):
        """
        Compute the weighted color similarity between two colors in HSV space.
        Uses the formula: Dr(Ri, Rj) = α|Hi - Hj| + β|Si - Sj| + γ|Vi - Vj|

        Parameters:
        - color1, color2: HSV color tuples
        - alpha: Weight coefficient for Hue (H) difference
        - beta: Weight coefficient for Saturation (S) difference
        - gamma: Weight coefficient for Value (V) difference

        Returns:
        - Weighted color distance
        """
        h1, s1, v1 = tuple(map(int, color1))
        h2, s2, v2 = tuple(map(int, color2))

        # For Hue, handle circular distance (e.g., distance between 170 and 10 is 20, not 160)
        h_diff = min(
            abs(h1 - h2), 180 - abs(h1 - h2) if max(h1, h2) > 90 else abs(h1 - h2)
        )
        s_diff = abs(s1 - s2)
        v_diff = abs(v1 - v2)

        # Apply weighted formula from the paper
        distance = alpha * h_diff + beta * s_diff + gamma * v_diff

        return distance

    def merge_similar_regions(
        self,
        quantized_img,
        skin_mask,
        similarity_threshold=20,
        alpha=5,
        beta=1,
        gamma=1,
        max_iterations=3,
    ):
        """
        Step 3 of Garcia-Tziritas algorithm: Merge adjacent skin color regions based on color similarity.

        The algorithm:
        1. First identifies all skin color regions and builds region adjacency graph
        2. For each pair of adjacent regions, checks if their colors are similar enough
        3. If similar, merges them into a larger region
        4. Limits merging process to a maximum of 3 iterations to reduce computational cost
        5. Returns all merged groups from each iteration

        Parameters:
        - quantized_img (numpy.ndarray): Quantized HSV image
        - skin_mask (numpy.ndarray): Binary mask where 255 indicates skin pixels
        - similarity_threshold (float): Threshold for color similarity
        - alpha: Weight coefficient for Hue (H) difference
        - beta: Weight coefficient for Saturation (S) difference
        - gamma: Weight coefficient for Value (V) difference
        - max_iterations: Maximum number of merging iterations (default: 3 as per paper)

        Returns:
        - all_merged_groups: List of merged groups from all iterations
        """
        # Get initial regions with their bounding boxes
        merged_regions = self.merge_adjacent_color_regions(quantized_img, skin_mask)

        # Keep track of merging iterations
        current_iteration = 0

        # Store all merged groups from each iteration
        all_merged_groups = []

        # Continue merging until we reach max iterations
        while current_iteration < max_iterations:
            # Create region adjacency graph
            # Keys are (color, region_id) tuples, values are sets of adjacent (color, region_id) tuples
            adjacency_graph = {}

            # Create a labeled image where each region has a unique identifier
            region_label_img = np.zeros(quantized_img.shape[:2], dtype=np.int32)

            # Assign unique label to each region and build region lookup by centroid
            region_by_centroid = {}
            next_label = 1

            for color, regions in merged_regions.items():
                for region in regions:
                    # Create unique identifier for this region
                    region_id = (color, next_label)
                    next_label += 1

                    # Store centroid lookup
                    centroid = tuple(map(int, region["centroid"]))
                    region_by_centroid[centroid] = region_id

                    # Fill region in label image
                    x, y, w, h = region["bbox"]
                    # Create mask for this specific region
                    color_array = np.array(color).reshape(1, 1, 3)
                    region_mask = np.all(
                        quantized_img[y : y + h, x : x + w] == color_array, axis=2
                    )
                    region_label_img[y : y + h, x : x + w][region_mask] = next_label - 1

                    # Initialize adjacency list
                    adjacency_graph[region_id] = set()

            # Find adjacent regions
            height, width = region_label_img.shape
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

            for y in range(height):
                for x in range(width):
                    if region_label_img[y, x] == 0:  # Skip background
                        continue

                    region1_label = region_label_img[y, x]
                    region1_color = tuple(quantized_img[y, x])
                    region1_id = None

                    # Find region ID from label
                    for color, regions in merged_regions.items():
                        if color == region1_color:
                            for region in regions:
                                # Check if this pixel is within the region's bbox
                                bbox_x, bbox_y, bbox_w, bbox_h = region["bbox"]
                                if (bbox_x <= x < bbox_x + bbox_w) and (
                                    bbox_y <= y < bbox_y + bbox_h
                                ):
                                    centroid = tuple(map(int, region["centroid"]))
                                    if centroid in region_by_centroid:
                                        region1_id = region_by_centroid[centroid]
                                        break
                            if region1_id:
                                break

                    if not region1_id:
                        continue

                    # Check neighbors
                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx

                        if not (0 <= ny < height and 0 <= nx < width):
                            continue

                        if region_label_img[ny, nx] == 0:  # Skip background
                            continue

                        region2_label = region_label_img[ny, nx]

                        if region2_label == region1_label:  # Same region
                            continue

                        region2_color = tuple(quantized_img[ny, nx])
                        region2_id = None

                        # Find region ID from label (similar to region1)
                        for color, regions in merged_regions.items():
                            if color == region2_color:
                                for region in regions:
                                    bbox_x, bbox_y, bbox_w, bbox_h = region["bbox"]
                                    if (bbox_x <= nx < bbox_x + bbox_w) and (
                                        bbox_y <= ny < bbox_y + bbox_h
                                    ):
                                        centroid = tuple(map(int, region["centroid"]))
                                        if centroid in region_by_centroid:
                                            region2_id = region_by_centroid[centroid]
                                            break
                                if region2_id:
                                    break

                        if not region2_id:
                            continue

                        # Add to adjacency graph
                        adjacency_graph[region1_id].add(region2_id)
                        adjacency_graph[region2_id].add(region1_id)

            # Compute color similarities and merge similar adjacent regions
            merged_groups = []
            visited = set()
            color_distances = []

            for region_id in adjacency_graph:
                if region_id in visited:
                    continue

                # Start a new merged group
                current_group = [region_id]
                visited.add(region_id)
                queue = [region_id]

                # BFS to find all similar connected regions
                while queue:
                    current_id = queue.pop(0)
                    current_color = current_id[0]  # Extract color from region_id tuple

                    for neighbor_id in adjacency_graph[current_id]:
                        if neighbor_id in visited:
                            continue

                        neighbor_color = neighbor_id[0]

                        # Compute weighted color similarity using our custom function
                        color_distance = self.color_similarity(
                            current_color, neighbor_color, alpha, beta, gamma
                        )
                        color_distances.append(color_distance)

                        # If colors are similar, merge regions
                        if color_distance <= similarity_threshold:
                            current_group.append(neighbor_id)
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)

                merged_groups.append(current_group)

            # Add current iteration's merged groups to all_merged_groups
            all_merged_groups.extend(merged_groups)

            # Check if we had any merges in this iteration
            total_merges_this_iteration = sum(
                len(group) - 1 for group in merged_groups if len(group) > 1
            )

            if self.debug:
                print(
                    f"Iteration {current_iteration+1}: Merged {total_merges_this_iteration} regions"
                )

            # If no merges occurred or this is the final iteration, we can return all results
            if (
                total_merges_this_iteration == 0
                or current_iteration == max_iterations - 1
            ):
                if self.debug:
                    print(f"Total iterations: {current_iteration+1}")
                    print(
                        f"Total merged groups across all iterations: {len(all_merged_groups)}"
                    )
                    if len(color_distances) > 0:
                        plt.figure(figsize=(10, 5))
                        plt.hist(color_distances, bins=20)
                        plt.title("Color distances distributions")
                        plt.xlabel("Color Distance")
                        plt.ylabel("Frequency")
                        plt.show()
                return all_merged_groups

            # Prepare for next iteration by updating regions
            # We need to create a new set of merged regions based on our groups
            new_merged_regions = {}

            for group in merged_groups:
                if len(group) == 1:
                    # Single region, keep as is
                    region_id = group[0]
                    color = region_id[0]

                    if color not in new_merged_regions:
                        new_merged_regions[color] = []

                    # Find the original region data
                    for region in merged_regions[color]:
                        centroid = tuple(map(int, region["centroid"]))
                        if region_by_centroid.get(centroid) == region_id:
                            new_merged_regions[color].append(region)
                            break
                else:
                    # Multiple regions to merge
                    # Use the color of the first region as the representative color
                    representative_color = group[0][0]

                    if representative_color not in new_merged_regions:
                        new_merged_regions[representative_color] = []

                    # Create a merged region from all regions in the group
                    bbox_mins = [float("inf"), float("inf")]
                    bbox_maxs = [0, 0]
                    total_area = 0
                    centroid_sum = [0, 0]

                    for region_id in group:
                        color, _ = region_id

                        # Find the original region data
                        for region in merged_regions[color]:
                            centroid = tuple(map(int, region["centroid"]))
                            if region_by_centroid.get(centroid) == region_id:
                                x, y, w, h = region["bbox"]
                                bbox_mins[0] = min(bbox_mins[0], x)
                                bbox_mins[1] = min(bbox_mins[1], y)
                                bbox_maxs[0] = max(bbox_maxs[0], x + w)
                                bbox_maxs[1] = max(bbox_maxs[1], y + h)

                                total_area += region["area"]
                                centroid_sum[0] += (
                                    region["centroid"][0] * region["area"]
                                )
                                centroid_sum[1] += (
                                    region["centroid"][1] * region["area"]
                                )
                                break

                    # Create the merged region
                    merged_region = {
                        "bbox": (
                            bbox_mins[0],
                            bbox_mins[1],
                            bbox_maxs[0] - bbox_mins[0],
                            bbox_maxs[1] - bbox_mins[1],
                        ),
                        "area": total_area,
                        "centroid": [
                            centroid_sum[0] / total_area,
                            centroid_sum[1] / total_area,
                        ],
                    }

                    new_merged_regions[representative_color].append(merged_region)

            # Update merged_regions for next iteration
            merged_regions = new_merged_regions
            current_iteration += 1

            if self.debug:
                print(
                    f"Regions after iteration {current_iteration}: {sum(len(regions) for regions in merged_regions.values())}"
                )

        # If we've reached max iterations, return all accumulated groups
        if self.debug:
            if len(color_distances) > 0:
                plt.figure(figsize=(10, 5))
                plt.hist(color_distances, bins=20)
                plt.title("Color distances distributions")
                plt.xlabel("Color Distance")
                plt.ylabel("Frequency")
                plt.show()

        return all_merged_groups

    def HSV_garcia(
        self,
        img,
        similarity_threshold=30,
        alpha=5,
        beta=0.2,
        gamma=0.2,
        max_iterations=3,
    ):
        """
        Creates a skin mask in the image using the HSV color space thresholding
        following the method described in the paper 'Face Detection Using Quantized
        Skin Color Regions Merging and Wavelet Packet Analysis' (Garcia and Tziritas, 1999).

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.
        - similarity_threshold (float): Threshold for merging similar color regions
        - alpha: Weight coefficient for Hue (H) difference
        - beta: Weight coefficient for Saturation (S) difference
        - gamma: Weight coefficient for Value (V) difference
        - max_iterations: Maximum number of merging iterations (default: 3 as per paper)

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """
        ## Step 1: Quantize the image using KMeans
        quantized = self.quantize(img, space="hsv")

        ## Step 2: Apply the HSV filter using values from the paper
        skin_mask_uint8 = self.HSV_filter_garcia(quantized)

        ## Step 3: Merge adjacent regions with similar colors using weighted color similarity
        merged_groups = self.merge_similar_regions(
            quantized,
            skin_mask_uint8,
            similarity_threshold=similarity_threshold,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_iterations=max_iterations,
        )

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

        # Calculate density map dimensions (M/4 × N/4 adapted from the paper)
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
                    # Keep 1 only if it has more than 2 neighbors with value 1
                    if neighbor_sum < 3:
                        filtered_mask[i, j] = 0
                else:
                    # Convert 0 to 1 if it has more than 3 neighbors with value 1
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
        if self.debug:
            plt.figure(figsize=(15, 9))
            masks = [
                img,
                color_mask,
                density_mask,
                luminance_mask,
                geometric_corrected_mask,
                final_mask,
            ]
            titles = [
                "Original Image",
                "Color Mask",
                "Density Mask",
                "Luminance Mask",
                "Geometric Corrected Mask",
                "Final Mask",
            ]
            for i, (mask, title) in enumerate(zip(masks, titles)):
                plt.subplot(2, 3, i + 1)
                plt.imshow(mask, cmap="gray")
                plt.title(title)
                plt.axis("off")
                if i == 2:
                    break
            for i, (mask, title) in enumerate(zip(masks[3:], titles[3:]), start=4):
                plt.subplot(2, 3, i)
                plt.imshow(mask, cmap="gray")
                plt.title(title)
                plt.axis("off")
            plt.show()

        return final_mask.astype(np.uint8) * 255

    def compute(self, img, method="naive_otsu"):
        methods = {
            "naive_otsu": self.naive_otsu,
            "ycrcb_garcia": self.YCrCb_garcia,
            "hsv_garcia": self.HSV_garcia,
            "ycrcb_chai": self.ycrcb_chai,
        }
        if method == "naive_otsu_quantized":
            method = "naive_otsu"
            args = {"quantized": True}
        else:
            args = {}
        try:
            return methods[method](img, **args)
        except KeyError:
            raise ValueError(f"Invalid method for skin mask: {method}")

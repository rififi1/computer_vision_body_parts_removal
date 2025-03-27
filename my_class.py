import numpy as np
import pandas as pd
from tqdm import tqdm
from skin_mask import SkinMask
import matplotlib.pyplot as plt
import cv2
import os
import json
import warnings

warnings.filterwarnings("ignore")


class HumanBodyPartNotFoundError(Exception):
    """Exception raised when no human body parts are detected in the image."""

    def __init__(self, message):
        super().__init__(message)


class NoHumanMaskError(Exception):
    """Exception raised when the human mask is empty."""

    def __init__(self, message):
        super().__init__(message)


class MyClass:
    def __init__(self, dataset_path, csv_path, with_edge_mask=False, min_area=100):
        """
        Initialize MyClass with dataset and CSV paths.

        Parameters:
        - dataset_path (str): Path to the dataset directory.
        - csv_path (str): Path to the CSV file containing polygon data.
        - min_area (int): Minimum area for human contours.
        """

        self.root_dir = os.getcwd()
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.min_area_for_human_contours = min_area
        self.with_edge_mask = with_edge_mask

        # loading csv file
        self.load_csv()

    def load_csv(self):
        """
        Load the CSV file and process the dataframe to extract polygon points.
        """

        # load csv
        csv_path = os.path.join(self.root_dir, self.csv_path)
        self.df_poly = pd.read_csv(csv_path)

        # extract the bounding polygon points
        self.df_poly["region_shape_attributes_json"] = self.df_poly[
            "region_shape_attributes"
        ].apply(json.loads)
        df_expanded = self.df_poly["region_shape_attributes_json"].apply(pd.Series)
        self.df_poly = self.df_poly.join(df_expanded)

        # filter out images without human body parts
        self.df_poly = self.df_poly[self.df_poly["all_points_x"].notna()]

        # keep only useful attributes
        self.df_poly = self.df_poly.drop(
            [
                "file_size",
                "file_attributes",
                "region_shape_attributes",
                "region_attributes",
                "region_shape_attributes_json",
                "name",
            ],
            axis=1,
        )

        # create a filenumber attribute for better file management
        self.df_poly["filenumber"] = (
            self.df_poly["filename"].rank(method="dense").astype(int)
        )

        # create a df specifically for filenames, filenumbers and region counts
        self.df_filenames = self.df_poly.drop_duplicates(
            ["filename", "filenumber", "region_count"]
        )[["filename", "filenumber", "region_count"]].set_index("filenumber")

    def load_img(self, filename):
        """
        Load an image from the dataset.

        Parameters:
        - filename (str): Name of the image file to load.

        Returns:
        - img (numpy.ndarray): Loaded image in RGB format.
        """

        filepath = os.path.join(self.dataset_path, filename)
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {filepath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show_images(self, images, titles, cmap=None):
        """
        Display a list of images with titles.

        Parameters:
        - images (list): List of images to display.
        - titles (list): List of titles corresponding to the images.
        - cmap (str): Colormap for the images.
        """

        plt.figure(figsize=(15, 12))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(3, 4, i + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(title)
            plt.axis("off")
        plt.show()

    def skin_detect(self, img, method="naive_otsu", debug=False):
        """
        Detect skin regions in the image.

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - skin_mask (numpy.ndarray): Binary mask of detected skin regions.
        """
        skin_mask = SkinMask(debug=False).compute(img, method=method)
        return skin_mask

    def edge_detection(self, img):
        """
        Detect edges in the image.

        Parameters:
        - img (numpy.ndarray): Input image in RGB format.

        Returns:
        - edges (numpy.ndarray): Binary edge map.
        """

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        return edges

    def combine_masks(self, skin_mask, edge_mask):
        """
        Combine skin mask and edge mask.

        Parameters:
        - skin_mask (numpy.ndarray): Binary mask of skin regions.
        - edge_mask (numpy.ndarray): Binary edge map.

        Returns:
        - combined_mask (numpy.ndarray): Combined binary mask.
        """

        combined_mask = (
            cv2.bitwise_or(skin_mask, edge_mask) if self.with_edge_mask else skin_mask
        )

        # TODO: I have a feeling we should only do opening here (= erosion + dilation)

        """# Closing = dilation + erosion
		kernel = np.ones((5, 5), np.uint8)
		combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
		# here we re-do dilation
		combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)"""
        return combined_mask

    def extract_human_contours(self, mask, min_area=100):
        """
        Extract human contours from the mask.

        Parameters:
        - mask (numpy.ndarray): Binary mask.
        - min_area (int): Minimum area for contours to be considered.

        Returns:
        - filtered_contours (list): List of filtered contours.
        """

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        return filtered_contours

    def create_mask(self, img, contours):
        """
        Create a mask from contours.

        Parameters:
        - img (numpy.ndarray): Input image.
        - contours (list): List of contours.

        Returns:
        - mask (numpy.ndarray): Binary mask created from contours.
        """

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask

    def refine_mask_with_grabcut(self, img, mask):
        """
        Refine the mask using GrabCut algorithm.

        Parameters:
        - img (numpy.ndarray): Input image.
        - mask (numpy.ndarray): Initial binary mask.

        Returns:
        - refined_mask (numpy.ndarray): Refined binary mask.
        """

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        mask_grabcut = np.where(mask == 255, cv2.GC_FGD, cv2.GC_BGD).astype("uint8")

        cv2.grabCut(
            img, mask_grabcut, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
        )

        refined_mask = np.where(
            (mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 255, 0
        ).astype("uint8")

        return refined_mask

    def inpaint_image(self, img, mask):
        """
        Inpaint the image using the mask.

        Parameters:
        - img (numpy.ndarray): Input image.
        - mask (numpy.ndarray): Binary mask for inpainting.

        Returns:
        - inpainted_img (numpy.ndarray): Inpainted image.
        """

        return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def detect_double_background(self, img, threshold=50):
        """
        Detect double background in the image.

        Parameters:
        - img (numpy.ndarray): Input image.
        - threshold (int): Threshold value for background detection.

        Returns:
        - bg_mask (numpy.ndarray): Binary mask of the detected background.
        """
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Threshold to find the uniform background color
        _, bg_mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        return bg_mask

    def filter_edges_with_background(self, img, edge_mask):
        """
        Filter edges by excluding the background.

        Parameters:
        - img (numpy.ndarray): Input image.
        - edge_mask (numpy.ndarray): Binary edge map.

        Returns:
        - combined_mask (numpy.ndarray): Edge mask with background excluded.
        """
        # Detect the background using color thresholding
        double_bg_mask = self.detect_double_background(img)

        self.double_bg_mask = double_bg_mask
        self.not_double_bg_mask = cv2.bitwise_not(double_bg_mask)

        # Combine edge mask and double background mask to exclude the background in the edge detection
        combined_mask = cv2.bitwise_and(edge_mask, cv2.bitwise_not(double_bg_mask))

        return combined_mask

    def compute_iou(self, refined_mask, filenumber, filename):
        """
        Compute the Intersection over Union (IoU) between the refined mask and ground truth.

        Parameters:
        - refined_mask (numpy.ndarray): The predicted mask from the algorithm.
        - filenumber (int): Index to identify which ground truth to use.
        - filename (str): Name of the file.

        Returns:
        - float: IoU score between 0 and 1
        """
        # Get ground truth mask
        gt_mask = self.get_ground_truth_mask(filenumber, filename, refined_mask.shape)

        # Ensure both masks are binary (0 or 255)
        refined_binary = (refined_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(refined_binary, gt_binary).sum()
        union = np.logical_or(refined_binary, gt_binary).sum()

        # Avoid division by zero
        if union == 0:
            return 0.0

        # Return IoU score
        return intersection / union

    def get_ground_truth_mask(self, filenumber, filename, shape):
        """
        Create a mask from the ground truth polygon coordinates in the dataframe.

        Parameters:
        - filenumber (int): Index to identify which ground truth to use.
        - filename (str): Name of the file.
        - shape (tuple): Shape of the image.

        Returns:
        -  gt_mask (numpy.ndarray): Binary mask representing the ground truth
        """
        number_regions = self.df_filenames.loc[filenumber, "region_count"]

        # Access the dataframe to get polygon coordinates
        x_points = self.df_poly.loc[
            self.df_poly["filenumber"] == filenumber, "all_points_x"
        ].tolist()
        y_points = self.df_poly.loc[
            self.df_poly["filenumber"] == filenumber, "all_points_y"
        ].tolist()

        # Create a list of lists of points from x and y coordinates
        points = []
        for i in range(number_regions):
            points.append(np.column_stack((x_points[i], y_points[i])).astype(np.int32))

        # Create an empty mask with the same size as the image
        mask_shape = shape[:2]
        gt_mask = np.zeros(mask_shape, dtype=np.uint8)

        # Fill the polygons on the mask
        for i in range(number_regions):
            cv2.fillPoly(gt_mask, [points[i]], 255)

        return gt_mask

    def get_filename(self, filenumber):
        """
        Get the filename associated with a filenumber.

        Parameters:
        - filenumber (int): Index to identify the filename.

        Returns:
        - str: Filename associated with the filenumber.
        """
        return self.df_filenames.loc[filenumber, "filename"]

    def compute_refined_mask(self, img, skin_detection_method="naive_otsu"):
        """
        Compute the refined mask for human body parts in the image.

        Parameters:
        - img (numpy.ndarray): Input image.

        Returns:
        - tuple: Various masks and contours used in the process.
        """

        # Skin detection
        skin_mask = self.skin_detect(img, method=skin_detection_method)
        # Edge detection (after filtering background edges)
        edges = self.edge_detection(img)
        filtered_edges = self.filter_edges_with_background(img, edges)

        # Combine masks
        combined_mask = self.combine_masks(skin_mask, filtered_edges)

        if skin_detection_method == "ycrcb_chai":
            ## Do not use contours for ycrcb_chai as it is detrimental to performance
            human_mask = combined_mask.astype(np.uint8) * 255
        else:
            contours = self.extract_human_contours(combined_mask)

            if not contours:
                print("no contours found")
                raise HumanBodyPartNotFoundError("No human body parts detected.")
            human_mask = self.create_mask(img, contours)

        if (not np.where(human_mask != 255)[0].any()) and (
            not np.where(human_mask != 255)[0].any()
        ):
            # Uncomment the line below to debug:
            # self.show_images([human_mask, combined_mask], ['Human Mask', 'Combined Mask'], cmap='gray')
            raise NoHumanMaskError(
                "Human Mask was empty, even though a body part was detected."
            )

        # Refine mask using GrabCut
        refined_mask = self.refine_mask_with_grabcut(img, human_mask)

        return (
            skin_mask,
            edges,
            filtered_edges,
            combined_mask,
            human_mask,
            refined_mask,
        )

    def remove_human_parts_with_visualization(
        self, filenumber, skin_detection_method="naive_otsu"
    ):
        """
        Pipeline: Remove human parts from the image and visualize the process.

        Parameters:
        - filenumber (int): Index to identify the image.

        Returns:
        - img_final (numpy.ndarray): Final inpainted image.
        """

        # get filename and load image from filenumber
        filename = self.get_filename(filenumber)
        img = self.load_img(filename)

        try:
            (
                skin_mask,
                edges,
                filtered_edges,
                combined_mask,
                human_mask,
                refined_mask,
            ) = self.compute_refined_mask(
                img, skin_detection_method=skin_detection_method
            )
        except HumanBodyPartNotFoundError as e:
            raise e
        except NoHumanMaskError as e:
            raise e
        except Exception as e:
            raise e

        print("IoU: ", self.compute_iou(refined_mask, filenumber, filename))

        ground_truth_mask = self.get_ground_truth_mask(
            filenumber, filename, refined_mask.shape
        )

        # Inpainting
        img_final = self.inpaint_image(img, refined_mask)
        # Visualization
        self.show_images(
            [
                img,
                skin_mask,
                edges,
                self.double_bg_mask,
                self.not_double_bg_mask,
                filtered_edges,
                combined_mask,
                human_mask,
                refined_mask,
                ground_truth_mask,
                img_final,
            ],
            [
                "Og Image",
                "Skin Mask",
                "Edge Mask",
                "double_bg_mask",
                "Not double_bg_mask",
                "Filtered Edge Mask",
                "Combined Mask",
                "Human Mask",
                "Refined Mask",
                "Ground Truth",
                "Final Inpainted Image",
            ],
            cmap="gray",
        )

        return img_final

    def mean_iou(self, skin_detection_method="naive_otsu"):
        """
        Compute the mean Intersection over Union (IoU) for all images.

        Returns:
        - float: Mean IoU score.
        """

        filenumbers = self.df_filenames.index.values
        total_iou = 0.0
        number_non_detected = 0

        for filenumber in tqdm(filenumbers, total=len(filenumbers)):
            filename = self.get_filename(filenumber)
            img = self.load_img(filename)
            try:
                _, _, _, _, _, refined_mask = self.compute_refined_mask(
                    img, skin_detection_method=skin_detection_method
                )
                total_iou += self.compute_iou(refined_mask, filenumber, filename)
            except Exception as e:
                print("filenumber: ", filenumber, " - filename: ", filename)
                print(e)
                number_non_detected += 1
        mean_iou = total_iou / (filenumbers.shape[0] - number_non_detected)
        print(f"without counting non detected human masks, mean IoU is: {mean_iou:.3f}")

        total_iou /= filenumbers.shape[0]

        return total_iou

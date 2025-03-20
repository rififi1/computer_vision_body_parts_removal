## Overview ✨
This project focuses on the removal of human body parts from images using traditional computer vision techniques. The approach includes dataset preprocessing, segmentation, mask generation, and inpainting for seamless removal.

## Repository Structure 📁
```
computer_vision_body_parts_removal/
│── data/                           
│   ├── Human_limbs_annotations.csv # Human limbs annotations (CSV format)
│   ├── CV_subset/                  # Folder containing image dataset
│── dev_notebooks/
│── main.ipynb                      # Main notebook for running the analysis
│── my_class.py                     # Class to remove human body parts
│── requirements.txt                # Needed dependencies
│── README.md                       # Project documentation
```

## Prerequisites 📋
### Dataset Requirements
To run the project, you will need:
- **CV_subset**: A folder containing image data.
- **Human_limbs_annotations.csv**: A CSV file with human limb annotations.

Ensure that the dataset is correctly placed in the `data/` directory before running the analysis.

## Quickstart 🚀
### 1. Download and Prepare Data
Place the `CV_subset` folder and `Human_limbs_annotations.csv` inside the `data/` directory.

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages using:

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
1. Open `main.ipynb` in Jupyter Notebook.
2. Run all the cells sequentially.
3. Adjust the `filenumber` parameter to inspect different images.


## Future Improvements 🔍
- Enhancing segmentation accuracy using advanced contour detection.
- Exploring additional inpainting techniques for better realism.
- Automating dataset preprocessing for scalability.

---

🔹 **Contributions & Feedback**: Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request. 🚀


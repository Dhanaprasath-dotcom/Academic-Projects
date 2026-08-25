# Image Compression with K-Means

## Project Overview

This educational computer-vision project compresses a cat image by reducing the number of colors with OpenCV's K-Means clustering algorithm. It compares the original image with the compressed result.

## Problem Statement

The project demonstrates how image colors can be grouped into a smaller number of clusters to reduce color complexity while preserving the general appearance of the image.

## Technologies Used

- Python
- OpenCV (`cv2`) for image loading, color conversion, K-Means clustering, and image writing
- NumPy for pixel-array reshaping and numeric data conversion
- Matplotlib for displaying the original and compressed images
- Jupyter Notebook for interactive execution
- `os` is imported in the notebook, but its documented role is not specified

## Features

- Load `cat.jpg` with OpenCV.
- Convert the image from BGR to RGB for display.
- Reshape image pixels into a two-dimensional array.
- Convert pixel data to 32-bit floating-point values for clustering.
- Group pixels into four color clusters using K-Means.
- Reconstruct the image from the cluster centers.
- Save the result as `compressed.jpg`.
- Display the original and compressed images side by side.

## Algorithm Configuration

- Number of color clusters: `k = 4`
- Termination criteria: epsilon plus a maximum of 100 iterations
- Epsilon: `0.2`
- K-Means attempts: `10`
- Center initialization: `cv2.KMEANS_RANDOM_CENTERS`

## Workflow

1. Load `cat.jpg` with OpenCV.
2. Convert the image from BGR to RGB.
3. Flatten the image into one row per pixel with three color channels.
4. Convert the pixel values to `float32`.
5. Apply K-Means clustering with four color centers.
6. Convert cluster centers to 8-bit unsigned integers.
7. Replace each pixel with the center of its assigned cluster.
8. Reshape the pixels back to the original image dimensions.
9. Save the reconstructed image as `compressed.jpg`.
10. Display the original and compressed images.

## Installation and Setup

### Requirements

- Python
- Jupyter Notebook
- OpenCV
- NumPy
- Matplotlib
- `cat.jpg` in the project folder

### Steps

1. Install Python and the required packages.
2. Place the input image at the project path as `cat.jpg`.
3. Open `imagecom.ipynb` in Jupyter Notebook or Visual Studio Code.
4. Run the notebook cells from top to bottom.

Exact installation commands and package versions: `TODO: Information required`

Environment variables: `TODO: Information required`

Configuration files: `TODO: Information required`

## Project Structure

```text
Model 9/
├── imagecom.ipynb
├── imagecom.md
├── cat.jpg
├── compressed.jpg
├── README.md
└── imagecom_files/
```

- `imagecom.ipynb`: Main image-compression notebook.
- `imagecom.md`: Markdown export of the notebook.
- `cat.jpg`: Input image.
- `compressed.jpg`: Generated compressed image.
- `README.md`: Project documentation.
- `imagecom_files/`: Notebook-related assets; detailed contents are not documented.

## Limitations and Missing Information

- The notebook does not measure file-size reduction or compression ratio.
- Image-quality metrics such as PSNR or SSIM are not documented.
- The output JPEG quality setting is not specified.
- The input image dimensions and file sizes are not documented.
- The result can vary because K-Means uses random center initialization.
- Package versions and exact installation commands are not documented.
- No license or image-source information is provided.

# Grain Boundary Detection with Stardist

The key folders/files in this project is:
- ModelTraining
    - TrainingModel_Results.ipynb
    - TrainingModel_Prediction.ipynb
    - trained_with_gpu.ipynb
    - combined_training.ipynb
    - pretrained_model.ip       ynb
- DataCollection
    - QuPathScript
    - KaggleDataProcessing
        - KaggleDataProcessing.ipynb
        - KaggleDataAnnotation.ipynb
    - BandContrastDataProcessing
        - BandContrastProcessing.ipynb
        - BandContrastAnnotation.ipynb
    - SAM Model.ipynb
    - dataConvexity.ipynb
- data
    - mydata
    - kaggleData
- KaggleGrainImages
    - Grains
    - Segemented
    - weights HED
    - HED
- Docs_Files
    - Stardist_DocsDemo_2D_Ben.ipynb
    -


# ModelTraining
##  TrainingModel_Results.ipynb
- Python environment with stardist library installed (pip install stardist).
- Image dataset in .tif format (Band Contrast images), with separate directories for images and masks (Refer to data folder)
- Data Loading: Automatically loads images and masks from specified directories.
- Preprocessing: Includes normalization and splitting into training and validation sets.
- Model Training: Utilizes a configurable StarDist model, with support for GPU acceleration (not implemented) and data augmentation.
- Evaluation: Offers threshold optimization and calculates various performance metrics.
- Visualization: Provides functions for visualizing the predictions alongside the actual images and masks.

##  TrainingModel_Prediction.ipynb
- Jupyter notebook designed for demonstrating the prediction capabilities of a pre-trained model (or personal loaded model) in grain boundary detection using the StarDist library. It focuses on predicting grain boundaries in Band Contrast images and visualizing the results.
- Data Loading: Loads images from `mydata` a specified directory for prediction.
- Image Preprocessing: Includes normalization of image channels, either independently or jointly.
- Model Configuration: Set up with specific parameters like the number of rays (n_rays), GPU usage (use_gpu), and grid settings.
- Model Importing: Demonstrates how to load a pre-trained StarDist model or a custom trained model from a zip file.
- Prediction: Normalizes input images and uses the model to predict grain boundaries, showcasing the results visually.
- 
## trained_with_gpu.ipynb
-shows how to use gpu with stardist for faster training

## combined_training.ipynb
- training multiple models of stardist, with one model focused on small grains and one focused on larger grains.
- creates combine images through post processing.

## pretrained_model.ipynb
- results of pretrained models on our data. deminstraits the problem of using a pre trained models as it does not preserve edges of grains.

# DataCollection
##  BandContrastDataProcessing
- Jupyter notebook designed for  preparing data for Band Contrast image analysis, focusing on creating cropped images and their corresponding masks from original Band Contrast image and manually annotated mask.
- Original Band Contrast images and it's mask(created via QuPath annotation) located in the `BC_Raw_Image`` folder.
- Crop Size Definition: Sets the crop dimensions (default 256x256) which can be adjusted as needed.
- Crop Calculation: Calculates the number of crops that fit into the original images based on their dimensions.
- Image Enhancement: Applies histogram equalization and contrast enhancement to the original image.
- Cropping Function: A function to crop both the original and mask images into the specified dimensions.
- Saving Cropped Images: The cropped images are saved into designated directories `mydata/train`.

## BandContrastAnnotation.ipynb
- Jupyter notebook designed for setting up the environment and generating masks for Band Contrast images using a pre-trained SAM model. 
`https://github.com/facebookresearch/segment-anything`
- Python environment with necessary libraries (torch, opencv-python, matplotlib, segment-anything).
- Directory structure for storing images and masks, specified in `dir_map`.
- Image Preprocessing: Performs HE and contrast enhancement on the full Band Contrast image.
- Cropping: Defines the crop size (attempted many including 100x100) and calculates the number of crops. It then crops the images and saves them to the specified directory.
- Model Download: Downloads the pre-trained SAM model from a specified URL.
- Mask Generation: Using the SAM model, generates masks for the cropped Band Contrast images.
- Mask Processing: Processes all images in the specified directory to generate and save masks.
- Annotation: Annotates the original images with generated masks and saves the annotated images in binary.

#  KaggleDataProcessing
## KaggleDataProcessing.ipynb
- This section describes the process of preparing a Kaggle dataset of images for use in grain boundary detection. `https://www.kaggle.com/code/peterwarren/multiple-edge-detection-methods/input`
- It involves converting images to grayscale, applying histogram equalization, and saving them in a specified directory. `kaggleData/train`

## KaggleDataAnnotation.ipynb
- Jupyter notebook designed to accomplish the same task as `BandContrastAnnotation.ipynb` above for the imported kaggle dataset.

##  SAM Model.ipynb
- Jupyter notebook designed to demo SAM model for use in grain segmentation for Band Contrast image used in training.

##  dataConvexity.ipynb
- Jupyter notebook designed to evaluate datasets (both Band Contrast and Kaggle images) for their suitability in grain boundary detection, focusing on the convexity of objects and the reconstruction of labeled images. This property of 'convexity' is required for Stardist to function properly. 
- Dataset Loading: Loads the Band Contrast and Kaggle images along with their masks.
- Convexity Testing: Uses a range of rays (2**i for i in range(2,8)) to fit ground-truth labels with star-convex polygons.
- Reconstruction Score Calculation: Calculates the mean intersection over union (IoU) for each number of rays to evaluate the accuracy of ground truth reconstruction.
- Plots are generated to visualize the reconstruction score against the number of rays used for both Band Contrast and Kaggle datasets.
- Example images are reconstructed with various numbers of rays to visually assess the reconstruction quality.
    - **Important Note:** The Kaggle dataset requires more preprocessing post-Segmentation Anything Model for mask creation due to objects not being sufficiently convex. The reconstruction score should ideally be greater than 0.8 for a reasonable number of rays to ensure dataset quality.

##  QuPathScript
- Script used to export mask annotated manually using QuPath
- Simply annotate the image and drag n' drop script into QuPath and directory/file structure will be created for you (Given you have vvcreated a project prior)

##  data
### mydata
- Holds the images and masks generated from `BandContrastDataProcessing.ipynb` using the Band Contrast image and the manually annotated mask via QuPath
- Used in Training the Stardist model

### kaggleData
- Holds the images and masks generated from `KaggleDataProcessing.ipynb` using the Kaggle dataset and annotated binary mask created via SAM 
- Was hoping to be used in Training the Stardist model
    - Kaggle dataset requires more preprocessing post-Segmentation Anything Model (refer to `dataConvexity.ipynb`)

##  KaggleGrainImages
### Grains
- Holds the original images from the Kaggle dataset 

### Segemented
- Holds the original masks from the Kaggle dataset 

### weights HED
- Holds the weights used in the HED model preciously employed by the owner of the Kaggle dataset

### HED
- Attempts to see results of this model mentioned above on Band Contrast image

##  Docs_Files
- Files taken from Stardist docs for reference
- Github: `https://github.com/maweigert/neubias_academy_stardist`
- Video: `https://www.youtube.com/watch?v=Amn_eHRGX5M`

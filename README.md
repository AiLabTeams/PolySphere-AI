# **Intelligent Perception and Regulation Algorithm Library for Polystyrene Microsphere Synthesis and Self-Assembly**

## **1.Project Introduction**

&nbsp;   This project open-sources a set of intelligent perception and regulation algorithms for polystyrene microsphere synthesis and gas-liquid interface self-assembly processes. The project includes four core modules, covering the entire process from microsphere synthesis particle size prediction, dispersion formulation optimization, self-assembly defect detection, to substrate arrangement quality assessment.



&nbsp;   This codebase aims to assist researchers in utilizing machine vision and deep learning technologies to achieve intelligent feedback and parameter regulation during microsphere preparation and assembly.



## **2.Directory Structure**

&nbsp;   This project consists of four subfolders corresponding to four different functional modules. Datasets for all modules are stored in the DataSet directory within their respective folders. The scripts are configured to load data automatically via relative paths, so no path modification is required for execution.



Project\_Root/

│

├─Polystyrene Microsphere Size Prediction

│			├── DNNBO/                          	 # Module 1:Polystyrene microsphere particle size prediction and formulation optimization

│			│   ├── DataSet/                   	 # Dataset directory (contains bodata.csv)

│			│   ├── AllModel.py                	 # Main training script and interactive prediction program

│			│   ├── Kfold.py                    	 # K-fold cross-validation performance testing script

│			│   └── dnn.py                     	 # Deep neural network model evaluation script

│			│

├─Polystyrene Substrate Grade Classification

│			├── classify/                      		 # Module 2:Polystyrene substrate arrangement grade classification

│			│   ├── DataSet/                    	 # Dataset directory (contains raw image and json data)

│			│   ├── json\_to\_csv.py             	 # Data preprocessing script (JSON to CSV and image segmentation)

│			│   ├── svmclassify.py              	 # Model training script (ResNet50 feature extraction + SVM)

│			│   └── classlogo.py               	 # Model application script (single image grade prediction and visualization)

│			│

├─Droplet Liquid Pit Opening/Closing Detection

│			├── pitDetection/                   	 # Module 3: Droplet-formed liquid pit opening/closing detection

│			│   ├── DataSet/                   	 # Dataset directory (contains labels.csv and images)

│			│   └── EfficientNet.py            	 # Training and testing script (EfficientNet-B0 + SVM)

│			│

└─Microsphere Dispersion Quality Coefficient Prediction

 			└── wt\_prediction/                  	# Module 4: Microsphere dispersion optimal assembly concentration (wt%) prediction

    			├── DataSet/                    		# Dataset directory (requires d1.csv)

   			└── design\_model.py        	# Polynomial regression model training and gradient correction script



## **3.Module Functions and Usage Instructions**

### *1. Polystyrene Microsphere Size Prediction (DNNBO)*

**Function Description:** This module uses a Deep Neural Network (DNN) to establish the mapping relationship between the dosage of synthesis reagents and the microsphere particle size. 



**Forward Prediction:** Input the dosages of four reagents to predict the synthesized microsphere particle size. Reverse Optimization: Combined with Bayesian Optimization (BO), it can deduce the experimental formula based on the target particle size.



**Reagent Parameter Description:**

KPS (Potassium Persulfate): 1% mass fraction aqueous solution (Unit: ml)

SDS (Sodium Dodecyl Sulfate): 1% mass fraction aqueous solution (Unit: ml)

H2O (Deionized Water): (Unit: ml)

C2H5OH (Ethanol): (Unit: ml)



**Script Usage:**

AllModel.py: The core script. Running it will train and save the model (my\_model.h5). The program provides an interactive command-line interface where users can enter 'KPS,SDS,H2O,C2H5OH,ST,type' to obtain prediction results.

Kfold.py \& dnn.py: Used for model performance validation in scientific papers, outputting cross-validation results and regression metrics (R², MAE, etc.), respectively.



### *2. Polystyrene Substrate Grade Classification (classify)*

**Function Description:** This module is used to evaluate the self-assembly arrangement quality (ordering) of microspheres on a substrate.



**Grade Definition:** The arrangement quality is divided into 4 grades (Class 1 - Class 4). Class 1 represents the highest grade (best ordering), and Class 4 represents the lowest grade (worst ordering).



**Script Usage:**

Data Preprocessing (json\_to\_csv.py):

&nbsp;       Input: Manually annotated raw images and corresponding .json segmentation files (located in DataSet/json/train/data).

&nbsp;       Operation: Running this script will crop small images based on the annotated regions and generate a labels.csv index file. This is a necessary step before training.

Model Training (svmclassify.py):

&nbsp;       Reads preprocessed data, uses pre-trained ResNet50 for feature extraction, and trains an SVM classifier.

&nbsp;       Outputs the model file svm\_classifier.joblib and feature extractor weights.

Model Application (classlogo.py):

&nbsp;       Loads the trained model, predicts a new single microscopic image, and outputs the grade region segmentation results for that image.



### *3. Droplet Liquid Pit Opening/Closing Detection (pitDetection)*

**Function Description:** This module performs binary classification on the morphology of "liquid pits" formed during droplet impact, determining whether they are "Open" or "Closed". 



**Class Labels:** 0 (Open) / 1 (Closed).



**Script Usage:**

EfficientNet.py: Integrates feature extraction and classification training.

Model Architecture: Uses EfficientNet-B0 as the feature extraction backbone network (automatically downloads pre-trained weights on the first run), connected to an SVM classifier at the backend.

Output: Generates the svm\_classifier\_efficientnet.joblib model file and outputs the ROC curve and confusion matrix to evaluate performance.



### *4. 微球分散液质量系数预测 (wt\_prediction)*

**Function Description:** This module is used to predict the optimal dispersion mass fraction (wt%) for microspheres of a specific particle size in self-assembly experiments. The algorithm uses a Polynomial Regression (Ridge/Lasso) model with gradient correction to improve fitting accuracy for non-linear data.



**Data Preparation:** Please ensure the source data file is named d1.csv and placed in the DataSet folder. The data column order must be strictly observed: \[size (particle size), cv (coefficient of variation), wt% (concentration), level (grade)].



**Script Usage:**

design\_model.py: Running this script will automatically execute the following process:

&nbsp;       Data cleaning and standardization.

&nbsp;       Model training (trained separately for different Level datasets).

&nbsp;       Generates visualization charts: prediction\_result.png (Predicted vs. Actual) and learning\_curve.png (Learning Curve).

&nbsp;       Saves the trained model files (.pkl) to the DataSet directory.



## **4.Environment Dependencies**

This project is primarily developed based on Python. Using an Anaconda environment is recommended. The main **dependency libraries** are as follows:



Python 3.12



PyTorch / torchvision (for feature extraction)



TensorFlow / Keras (for DNN models)



Scikit-learn (for SVM, regression, and data processing)



OpenCV (cv2)



Pandas, NumPy



Matplotlib, Seaborn (for plotting)



## Remarks

All algorithm modules are configured with relative paths. Please keep the internal folder structure (especially the location of the DataSet folder) unchanged to ensure the code can read data correctly.


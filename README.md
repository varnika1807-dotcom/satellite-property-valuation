# satellite-property-valuation
Multimodal regression for property price prediction using tabular data and satellite imagery
Project Overview

This project predicts house prices using a multimodal machine learning approach.
It combines:

Tabular property data such as size, rooms, condition, and location

Satellite images that capture neighborhood and environmental context

The goal is to show how visual information from satellite images can support traditional tabular features and improve model understanding in real estate valuation.

Approach Summary

The project is implemented in two main stages.

1. Tabular Baseline Model

A regression model is trained using only structured numerical features such as:

Bedrooms and bathrooms

Living area and lot size

Condition, grade, and view

Latitude and longitude

This model serves as a baseline for comparison.

2. Multimodal Model (Tabular + Satellite Images)

Satellite images are generated using latitude and longitude.
A pretrained convolutional neural network (ResNet) is used to extract image features.
These image features are combined with tabular features and used to train a regression model.
Grad-CAM is applied to explain which regions of the satellite images influence the model.

Repository Structure
├── cdc.ipynb              # Complete end-to-end project notebook
├── images/
│   └── train_500/      # Satellite images used for multimodal modeling
├── prediction.csv         # Final predictions on test data
└── README.md              # Project documentation

Notebook Description (cdc.ipynb)

The notebook contains the complete pipeline, clearly divided into sections:

Data loading and inspection

Exploratory Data Analysis (EDA)

Feature engineering

Tabular baseline model training

Satellite image generation

Image preprocessing and CNN feature extraction

Feature fusion (tabular and image features)

Multimodal model training and evaluation

Grad-CAM based explainability

Final prediction generation for test data

Each step is explained in simple language inside the notebook.

Dataset
Tabular Data

Includes features such as:

Bedrooms, bathrooms

Living area and lot size

Floors, condition, grade

Waterfront and view

Latitude and longitude

Neighbor-based features

Additional engineered features include:

Sale year and month

House age

Renovation indicator

Land utilization ratio

Visual Data

Satellite images are fetched using geographic coordinates.
The images capture neighborhood layout, greenery, and surrounding infrastructure.

How to Run the Project
1. Install Required Libraries
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision pillow opencv-python tqdm

2. Open and Run the Notebook

Open the file:

cdc.ipynb


Run the notebook from top to bottom.
All steps are executed in sequence.

3. Generate Predictions

The notebook generates a file named:

submission.csv


Format:

id,predicted_price

Results Summary
Results Summary
Model	RMSE	R² Score
Tabular Baseline	~113,767	~0.897
Multimodal (Tabular + Images)	~187,897	~0.721

The tabular model performs better in terms of numerical accuracy.
The multimodal model adds visual context and improves interpretability by using satellite images.

Explainability (Grad-CAM)

Grad-CAM is used to visualize which regions of the satellite images influence the CNN feature extraction.
The visualizations show attention over residential areas, green spaces, and overall neighborhood layout.
This confirms that the model captures high-level environmental context.

Limitations

Limited number of satellite images

CNN is used as a fixed feature extractor

No fine-tuning of image model

Satellite image resolution is constrained by API limitations

Future Scope

Fine-tune CNN on real estate imagery

Use higher-resolution satellite images

Try advanced feature fusion techniques

Include more neighborhood-level metadata

Note on Implementation

For simplicity and reproducibility, the entire project is implemented inside a single Jupyter notebook (cdc.ipynb).
Each stage of the pipeline is clearly separated and documented.

Author

This project was developed as part of a data science and multimodal machine learning assignment,
demonstrating the integration of computer vision and tabular modeling for real-world applications.

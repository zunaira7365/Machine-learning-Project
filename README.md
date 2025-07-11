# Machine-learning-Project:
This repository prepares a cleaned, feature-rich dataset and predictive models to estimate Airbnb listing prices in Berlin using multiple data modalities.

## Project Overview:

The goal is to predict Airbnb listing prices (price) by leveraging a rich feature set comprising:

Tabular features: Listing attributes like number of rooms, amenities, etc.

Spatial feature: Distance to city center (dist_to_center_km).

Image features: Extracted features from listing photos (2048-dimensional vectors).

The target variable is transformed using the natural logarithm to stabilize variance and improve model performance.

### 1. Load & Clean Listings + Calendar Data:

You can find and download all data files for all cities by clicking following link https://insideairbnb.com/get-the-data/.

Load listings.csv and clean the price column.

Load calendar.csv and compute the average price per listing.

Merge both into a base dataset by using Id from listing and Id_listing form Calendar file.

Parse and binarize top 17 amenities.

Save intermediate base_df.

### 2. Tabular Features:

Select tabular features: 
        e.g., bedrooms, bathrooms, review_scores_rating, instant_bookable, and amenities.

Binary features (e.g., instant bookable, amenity presence).

Missing values are filled with median values.

### 3. Spatial Features:

Compute distance of each listing to the Berlin city center using latitude & longitude via geopy.distance.geodesic.

Feature: dist_to_center_km.

Save intermediate features_df.

### 4. Image Downloading and Embeddings:

Download main listing images using picture_url for each listing using requests.

Load Keras ResNet50 (ImageNet weights, include_top=False).

Extract deep feature embeddings using ResNet50 (pretrained on ImageNet, Keras version).

Create 2048-dimensional vector per image.

Save to image_features.csv for future use.

### 5. Merge All Features into Final Dataset:

Merge:

Tabular, Spatial features and Image embeddings.

Drop latitude and longitude cloumns because we have created a new column of dist_to_center_km
by using them. So we don't need any more.

Place id as first column.

Save full dataset as cleaned_listings.csv.

### 6. Modelling preparations:

Import all important libraries.
Load the cleaned dataset (cleaned_listings.csv).

Filter out price outliers (keep listings priced between 20 and 500).

Create a log-transformed target variable (log_price).

Identify feature subsets:

Tabular + Spatial features: Exclude image columns and target columns.

Image features: Columns starting with image_feat_.

Split the dataset into training and test sets with an 80-20 ratio.

Perform feature selection on tabular + spatial features using SelectKBest with f_regression.

Scale tabular + spatial features using StandardScaler.

Scale raw image features and reduce dimensionality using PCA (1000 components).

Combine scaled tabular + spatial features with PCA-reduced image features.

Apply final scaling to combined features before modeling.

### 7. Modeling:

#### LightGBM Regressor:

Train a LightGBM model with default hyperparameters on combined features.

Evaluate predictions on test data, converting log predictions back to the original price scale.

Metrics reported: MAE, RMSE, R², MAPE, and accuracy (100 - MAPE).

Visualize predicted vs. actual prices.

**MAE: 21.94, RMSE: 40.44, R²: 0.495
MAPE: 21.44%, Accuracy: 78.56%**

#### Hyperparameter Tuning for LightGBM:

Perform RandomizedSearchCV to find better hyperparameters for LightGBM.

Tune parameters like num_leaves, max_depth, learning_rate, n_estimators, and regularization terms.

Re-evaluate the tuned model on the test set.

Output best parameters and evaluation metrics.

**Tuned MAE: 21.71, RMSE: 40.66, Accuracy: 78.83%**

#### Neural Network:

Define a simple feedforward neural network using TensorFlow/Keras:

Input layer matching the combined feature dimension.

Two hidden dense layers with ReLU activation and dropout for regularization.

Output layer for price prediction.

Compile model with adam optimizer and mean squared error loss.

Train with early stopping based on validation loss.

Predict on test set and evaluate with the same metrics as LightGBM.

Plot training and validation loss curves.

Visualize predicted vs. actual prices.

**Neural Network → MAE: 33.14, RMSE: 52.35, R²: 0.154
MAPE: 32.24%, Accuracy: 67.76%**
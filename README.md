# Machine-learning-Project:
This repository prepares a cleaned, feature-rich dataset for predicting Airbnb prices in Berlin. It includes tabular, spatial, and image features from publicly available Airbnb datasets.

### 1. Load & Clean Listings + Calendar Data:

you can download data for all cities by clicking following link
https://insideairbnb.com/get-the-data/

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

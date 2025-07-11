# Machine-learning-Project
This repository prepares a cleaned, feature-rich dataset for predicting Airbnb prices in Berlin. It includes tabular, spatial, and image features from publicly available Airbnb datasets.

#### Load & Clean Listings + Calendar Data

Load listings.csv and clean the price column.

Load calendar.csv and compute the average price per listing.

Merge both into a base dataset by using Id from listing and Id_listing form Calendar file.

Parse and binarize top 17 amenities.

Save intermediate base_df.
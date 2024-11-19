#!/bin/bash

# Create the target directories
mkdir -p downsampled_train downsampled_val downsampled_test

# Count the total number of images in the data directory
total_images=$(ls downsampled_data | wc -l)

# Calculate the number of images for each set
train_count=$((total_images * 80 / 100))
val_count=$((total_images * 10 / 100))
test_count=$((total_images - train_count - val_count))

# Shuffle the images and distribute them
shuf -e downsampled_data/* | {
    # Move images to the train set
    counter=0
    while [ $counter -lt $train_count ]; do
        read file
        mv "$file" downsampled_train/
        ((counter++))
    done

    # Move images to the val set
    counter=0
    while [ $counter -lt $val_count ]; do
        read file
        mv "$file" downsampled_val/
        ((counter++))
    done

    # Move remaining images to the test set
    while read file; do
        mv "$file" downsampled_test/
    done
}

#!/bin/bash

# Set the input and output directories
img_format="png"
input_dir="combined"
output_dir="downsampled_flares_v2"
res_h=353
res_w=263

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all jpg or png files in the "<input_dir>" folder
for image in "$input_dir"/*.$img_format; do
    # Extract the base filename without the extension
    base_name=$(basename "$image" .$img_format)

    # Set the output filename with "downsampled" postfix and the specified format
    output_image="$output_dir/${base_name}_downsampled.$img_format"

    # Downsample the image and copy it to the "downsampled" folder with the manual resolution
    ffmpeg -i "$image" -vf scale="${res_h}:${res_w}" "$output_image" -loglevel quiet

    echo "Processed: $image -> $output_image"
done

echo "All images have been downsampled and copied."

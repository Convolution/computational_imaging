# Create the 'data' directory if it doesn't exist
mkdir -p data

# Initialize the counter
counter=1

# Copy and rename images from 'reflection_layer'
for file in reflection_layer/*; do
    mv "$file" data/img_$(printf "%05d" "$counter").jpg
    ((counter++))
done

# Copy and rename images from 'transmission_layer'
for file in transmission_layer/*; do
    mv "$file" data/img_$(printf "%05d" "$counter").jpg
    ((counter++))
done

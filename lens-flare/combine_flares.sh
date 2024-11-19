# Create the 'combined' directory if it doesn't exist
mkdir -p combined

# Initialize the counter
counter=1

# Copy and rename images from 'captured'
for file in captured/*; do
    cp "$file" combined/flare_$(printf "%04d" "$counter").png
    ((counter++))
done

# Copy and rename images from 'simulated'
for file in simulated/*; do
    cp "$file" combined/flare_$(printf "%04d" "$counter").png
    ((counter++))
done

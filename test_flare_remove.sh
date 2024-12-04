srun -p csc2529 -c 8 --nodelist=squid06 --gres gpu:1 --pty \
python3 -m flare_removal.python.remove_flare \
--ckpt=flare_removal/trained_models/unet_3plus_entire_data/logs  \
--model='unet_3plus_2d' \
--input_dir=flare_free/real/downsampled_input \
--out_dir=flare_removal/model_tests/unet_3plus_entire_data/test_real

python3 -m flare_removal.python.calculate_metrics \
--gt_dir=flare_free/real/downsampled_ground_truth \
--blended_dir=flare_removal/model_tests/unet_3plus_entire_data/test_real/output_blend \
--out_dir=flare_removal/model_tests/unet_3plus_entire_data/test_real

srun -p csc2529 -c 8 --nodelist=squid06 --gres gpu:1 --pty \
python3 -m flare_removal.python.remove_flare \
--ckpt=flare_removal/trained_models/unet_3plus_entire_data/logs  \
--model='unet_3plus_2d' \
--input_dir=flare_free/synthetic/downsampled_input \
--out_dir=flare_removal/model_tests/unet_3plus_entire_data/test_synthetic

python3 -m flare_removal.python.calculate_metrics \
--gt_dir=flare_free/synthetic/downsampled_ground_truth \
--blended_dir=flare_removal/model_tests/unet_3plus_entire_data/test_synthetic/output_blend \
--out_dir=flare_removal/model_tests/unet_3plus_entire_data/test_synthetic

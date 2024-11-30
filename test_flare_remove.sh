srun -p csc2529 -c 8 --nodelist=squid06 --gres gpu:1 --pty \
python3 -m flare_removal.python.remove_flare \
--ckpt=flare_removal/unet_3plus_entire_data_training/logs  \
--model='unet_3plus_2d' \
--input_dir=flare_free/real/downsampled_input \
--out_dir=flare_removal/output_unet_3plus/test_real

srun -p csc2529 -c 8 --nodelist=squid06 --gres gpu:1 --pty \
python3 -m flare_removal.python.remove_flare \
--ckpt=flare_removal/unet_3plus_entire_data_training/logs  \
--model='unet_3plus_2d' \
--input_dir=flare_free/synthetic/downsampled_input \
--out_dir=flare_removal/output_unet_3plus/test_synthetic

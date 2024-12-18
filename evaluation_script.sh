srun -p csc2529 -c 8 --nodelist=squid03 --gres gpu:1 --pty python3 -m flare_removal.python.evaluate \
--eval_dir=flare_removal/unet_3plus_entire_data_eval \
--train_dir=flare_removal/unet_3plus_entire_data_training/logs \
--scene_dir=flare_free/real/downsampled_input \
--flare_dir=lens-flare/downsampled_flares_v2 \
--training_res=224 \
--flare_res_h=353 \
--flare_res_w=263 \
--model=unet_3plus_2d

srun -p csc2529 -c 8 --nodelist=squid03 --gres gpu:1 --pty python3 -m flare_removal.python.evaluate \
--eval_dir=flare_removal/unet_3plus_entire_data_eval \
--train_dir=flare_removal/unet_3plus_entire_data_training/logs \
--scene_dir=flare_free/synthetic/downsampled_input \
--flare_dir=lens-flare/downsampled_flares_v2 \
--training_res=224 \
--flare_res_h=353 \
--flare_res_w=263 \
--model=unet_3plus_2d

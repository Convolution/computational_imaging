srun -p csc2529 -c 8 --nodelist=coral02 --gres gpu:1 --pty python3 -m flare_removal.python.evaluate \

--eval_dir=flare_removal/evaluation \
--train_dir=flare_removal/training/logs \
--scene_dir=flare_free/downsampled_train \
--flare_dir=lens-flare/downsampled_flares \

# Warning: verify the resolution of the scene and flare images with 'file <image file name>'
--training_res=256 \ # scene image resolution (training_res X training_res)
--flare_res_h=402 \ # flare image hight resolution
--flare_res_w=300 # flare image width resolution


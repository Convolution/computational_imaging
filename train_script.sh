srun -p csc2529 -c 8 --nodelist=coral02 --gres gpu:1 --pty python3 -m flare_removal.python.train \

--train_dir=flare_removal/training/logs \
--scene_dir=flare_free/downsampled_train \
--flare_dir=lens-flare/downsampled_flares \
--epochs=100 \
--batch_size=2 \
--learning_rate=1e-4 \

# Warning: verify the resolution of the scene and flare images with 'file <image file name>'
--training_res=256 \ # scene image resolution (training_res X training_res)
--flare_res_h=402 \ # flare image hight resolution
--flare_res_w=300 # flare image width resolution

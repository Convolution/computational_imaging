# computational_imaging
CSC2529 project

Data preparation:
1) Go to flare_free and specify in downsample.sh the input dir (entire original size data - 27,449 images), output dir, and verify res is 224.
2) Run downsample.sh.
3) Download the real and synthetic test datasets from https://drive.google.com/drive/folders/1_gi3W8fOEusfmglJdiKCUwk3IA7B2jfQ and place them in flare_free.
4) Repeat steps 1 and 2 on ground_truth and input directories that are in real (20 images) and synthetic (37 images) directories.
5) Go to lens-flare and specify in downsample.sh the input dir (entire original flares data - 5,001 images), output dir, and verify that res_h and res_w are 353 and 263 respectively.
6) Run downsample.sh.


Model training:
1) Go to google-research and specify in train_script.sh train_dir (where to save logs), scene_dir (where to get entire downsampled data), flare_dir (where to get entire downsampled flares), epochs=200, training_res=224, flare_res_h=353, flare_res_w=263, model (name of the model you are running) and exp_name (_<name of the experiment>).
2) Run train_script.sh


Evaluating model (optional - if there is a separate evaluation data):
1) After model finished training go to google-research and specify in evaluation_script.sh eval_dir (where to save results), train_dir (logs dir of the trained model), scene_dir (path to downsampled ground_truth test dir, once for real and once for synthetic), training_res=224, flare_res_h=353, flare_res_w=263, model (name of the model type you are evaluating).
2) Run evaluation_script.sh.


Test model to produce visual and qualitative results:
1) In google-research specify in test_flare_remove.sh:
  a) for the commands starting with srun: ckpt (path to logs dir of trained model, model (model type), input_dir (path to downsampled input test dir, once for real and once for synthetic), out_dir (path where results are saved).
  b) for the commands starting with python3: gt_dir (path to ground truth images), blended_dir (path to blended images dir) and out_dir (output path of the text file with metrics).
2) Run test_flare_remove.sh.

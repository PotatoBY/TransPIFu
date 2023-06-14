## 1. Requirements

We provide a conda `yml` environment file (you can modify `prefix` to change installation location). This conda env. is for `TransPIFu`,`Geo-PIFu` and `PIFu` training, test and evaluation.

	conda env create -f transpifu_environment.yml
	conda activate transpifu

We use `opendr` for mesh rendering. To accomodate its package requirements, we provide another conda `yml` environment file. If you have trouble with `yml` we also provide our installation commands log in `assests/create_opendrEnv.sh`.

	conda env create -f opendr_requirements.yml
	conda activate opendrEnv # plz make sure this works
	conda deactivate         # back to the transpifu env.

## 2. Dataset

Download the `DeepHuman` mesh dataset from [here](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset) and move it into your preferred data folder. The downloaded data should be a zip file about 30.5 G.

	mv DeepHumanDataset.zip data/DeepHumanDataset/
	cd data/DeepHumanDataset
	unzip DeepHumanDataset.zip # plz visit https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset for password

Download the `LSUN` background image dataset and unzip the `zip` files.

	cd data/LSUN
	python3 TransPIFu/LSUN/download.py
	unzip "*.zip"

Parse the `lmdb` files into jpeg images. You might need to `pip install lmdb`.

	python TransPIFu/LSUN/data.py export ./*_val_lmdb --out_dir ./val_jpg --imageType jpg
	python TransPIFu/LSUN/data.py export ./*_train_lmdb --out_dir ./train_jpg --imageType jpg --limit 30000

## 3. Human Mesh Rendering and Voxelization

Please first activate the `opendrEnv` conda environment.
	
	conda activate opendrEnv

Generate SMPL-to-IUV mappings. This is for rendering `smplIUV` from parametric SMPL models. You might need `pip install pathlib` and `pip install tqdm`.

	cd TransPIFu/UVTextureConverter
	python obtain_facesAtlas_Idx_U_V.py

Compile the voxelizer cpp code. This is for voxelizing human meshes.

	cd TransPIFu/voxelizer
	mkdir build && cd build
	cmake ..
	make

You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple rendering scripts in parallel. For example, with 30 splits the whole rendering process will take about 37 hrs and generate 1.4T rendered data. The rendering script generates various items besides the color human images: `config`, `rgbImage`, `normalRGB`, `maskImage`, `meshVoxels`, `maskFromVoxels`, `meshSem`, `skeleton3D`, `skeleton2D`, `smplSem`, `smplIUV`, `smplSemVoxels`. Not all the items are needed/used in this project. But they could be very useful for other relevant tasks. Please read `render_mesh.py` for detailed explanation of each item and modify the rendering script accordingly to fit your need.

	cd TransPIFu/
	python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --splitNum 30 --splitIdx 0
	python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --splitNum 30 --splitIdx 1
	... # until --splitIdx 29

Instead of rendering the data on your own, we suggest using [the provided `config` files](https://www.dropbox.com/s/kxx87kyfuewhx0i/config_split_001_000.zip?dl=0) in order to obtain the same set of images used in our work `TransPIFu`. For each rendered image, we recorded its rendering settings like camera angles, lighting directions and so on in `json`.

	unzip config_split_001_000.zip -d data/humanRender/config/ # should expect to see 108718 json files

We provide a demo script of parsing these recorded `json` rendering setting files. You might need to make a few simple modifications in `render_mesh.py` in order to properly launch the script and fully re-render our data for fair experiment comparisons. As an alternative, we want to directly provide a download link of our rendered data. But this requires some agreements with the `DeepHuman` dataset authors. We are working on it.

    python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --useConfig --addionalType smplSemVoxels --splitNum 30 --splitIdx 0
    python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --useConfig --addionalType smplSemVoxels --splitNum 30 --splitIdx 1
    ... # until --splitIdx 29

## 4. Query Points Offline Sampling

Please first activate the `transpifu` conda environment.

	conda activate transpifu
	cd TransPIFu/transpifu

Prepare and save shape training query samples offline, because `--online_sampling` of `train_shape_iccv.py` is slow due to mesh reading, point sampling, ray tracing, etc. This query points offline sampling process only need to be done once. The sampled / saved query points can be used for `PIFu`,`Geo-PIFu` and `TransPIFu` training.

You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple sampling scripts in parallel. For example, with 32 splits the whole sampling process will take about 4 hrs and generate 146.5 G query points in total: saved at {`./occu_sigma3.5_pts5k_split32_00`, ..., `./occu_sigma3.5_pts5k_split32_31`}. We want to directly provide a download link of our sampled query points. But this requires some agreements with the `DeepHuman` dataset authors. We are working on it.

	python -m apps.prepare_shape_query --sampleType occu_sigma3.5_pts5k --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --epoch_range 0 15 --sigma 3.5 --num_sample_inout 5000 --num_sample_color 0 --splitNum 32 --splitIdx 0
    python -m apps.prepare_shape_query --sampleType occu_sigma3.5_pts5k --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --epoch_range 0 15 --sigma 3.5 --num_sample_inout 5000 --num_sample_color 0 --splitNum 32 --splitIdx 1
    ... # until --splitIdx 31

Zip and move the sampled query points from {`./occu_sigma3.5_pts5k_split32_00`, ..., `./occu_sigma3.5_pts5k_split32_31`} into the target data folder `${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k/`.

	mkdir ${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k
	cd ./occu_sigma3.5_pts5k_split32_00/ && find . -name '*.npy' -print | zip ../occu_sigma3.5_pts5k_split32_00.zip -@ && mv ../occu_sigma3.5_pts5k_split32_00.zip ${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k/ && rm *.npy
    cd ./occu_sigma3.5_pts5k_split32_01/ && find . -name '*.npy' -print | zip ../occu_sigma3.5_pts5k_split32_01.zip -@ && mv ../occu_sigma3.5_pts5k_split32_01.zip ${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k/ && rm *.npy
    ... # until occu_sigma3.5_pts5k_split32_31

Unzip these `zip` files inside `${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k/`.

	cd ${PREFERRED_DATA_FOLDER}/data/humanRender/occu_sigma3.5_pts5k
	unzip occu_sigma3.5_pts5k_split32_00.zip && rm occu_sigma3.5_pts5k_split32_00.zip
	unzip occu_sigma3.5_pts5k_split32_01.zip && rm occu_sigma3.5_pts5k_split32_01.zip
	... # until occu_sigma3.5_pts5k_split32_31

## 5. Training Scripts and Pre-trained Models

Training script for the `PIFu` baseline using the rendered DeepHuman images. 

	conda activate transpifu && cd TransPIFu/transpifu
	python -m apps.train_shape_iccv --gpu_ids 0 --name PIFu_baseline --sigma 3.5 --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --random_multiview --num_views 1 --batch_size 8 --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 888 --freq_save_ply 888 --z_size 200. --num_threads 8 # ~ 1 day

Download [our pre-trained weights of the `PIFu` baseline](https://www.dropbox.com/s/bc9du1zd2p2cqw8/netG_epoch_44_2415?dl=0) into the folder created below.

	mkdir TransPIFu/transpifu/checkpoints/PIFu_baseline # move the downloaded netG_epoch_44_2415 into this folder

Training scripts for `Geo-PIFu`. We adopt a staged training scheme of the coarse occupancy volume loss and the high-resolution query point loss. The second script below is to prepare aligned-latent-voxels for learning the final implicit function. You can modify `--deepVoxels_fusion` in the third script to play with different fusion schemes of the latent geometry and pixel features. By default we adopt early fusion, as explained in the paper.

    python -m apps.train_shape_coarse --gpu_ids 0 --name GeoPIFu_coarse --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --random_multiview --num_views 1 --batch_size 30 --num_epoch 30 --schedule 8 23 --freq_plot 1 --freq_save 970 --freq_save_ply 970 --num_threads 8 --num_sample_inout 0 --num_sample_color 0 --load_single_view_meshVoxels --vrn_occupancy_loss_type ce --weight_occu 1000.0 # ~ 2 days
    python -m apps.test_shape_coarse --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/geocoarsepifuResults/GeoPIFu_coarse/train --splitNum 1 --splitIdx 0 --gpu_id 0 --load_netV_checkpoint_path ./checkpoints/GeoPIFu_coarse/netV_epoch_29_2899 --load_from_multi_GPU_shape --dataType train --batch_size 1 # ~ 3 hrs, you can modify "splitNum" and "splitIdx" as used before for even more speedup
	python -m apps.train_query --gpu_ids 0 --name GeoPIFu_query --sigma 3.5 --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --deepVoxelsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/geocoarsepifuResults/GeoPIFu_coarse/train --random_multiview --num_views 1 --batch_size 1 --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 888 --freq_save_ply 888 --z_size 200. --num_threads 8 --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels # ~ 1 day

Download [our pre-trained weights of `Geo-PIFu`](https://www.dropbox.com/s/yln7blfxodhmbxh/geopifu_weights.zip?dl=0). Unzip the downloaded `geopifu_weights.zip` and put the weights into the two folders created below.

	mkdir TransPIFu/transpifu/checkpoints/GeoPIFu_coarse # move netV_epoch_29_2899 here
	mkdir TransPIFu/transpifu/checkpoints/GeoPIFu_query  # move netG_epoch_44_2415 here

Training script for the `TransPIFu` baseline using the rendered DeepHuman images. 

	conda activate transpifu && cd TransPIFu/transpifu
	python -m apps.train_shape_trans --gpu_ids 0 --name PIFu_transformer --sigma 3.5 --meshDirSearch /${PREFERRED_DATA_FOLDER}/data --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --random_multiview --num_views 1 --batch_size 2 --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 888 --freq_save_ply 888 --z_size 200. --num_threads 8 --normal_activate --depth_activate # ~ 1 day


## 6. Test Scripts

Test the models on the rendered 21744 DeepHuman test images. The test dataset is quite large and therefore you need at least 85 G to save the generated human meshes, which would be used for computing the evaluation metrics later. You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple test scripts in parallel. For example, with 7 splits the whole inference process will take about 3 hrs.

The scripts below are for `TransPIFu`.

    CUDA_VISIBLE_DEVICES=0 python -m apps.test_shape_iccv --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --splitNum 1 --splitIdx 0 --gpu_id 0 --load_netG_checkpoint_path ./checkpoints/PIFu_transformer/netG_epoch_43_3199 --name PIFu_transformer --normal_activate --depth_activate

The scripts below are for `Geo-PIFu`.

    CUDA_VISIBLE_DEVICES=0 python -m apps.test_shape_iccv --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --deepVoxelsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/geocoarsepifuResults/GeoPIFu_coarse/train --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --splitNum 1 --splitIdx 0 --gpu_id 0 --load_netG_checkpoint_path ./checkpoints/GeoPIFu_query/netG_epoch_44_2415

The scripts below are for `PIFu`.

    CUDA_VISIBLE_DEVICES=0 python -m apps.test_shape_iccv --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/PIFu_baseline --splitNum 1 --splitIdx 0 --gpu_id 0 --load_netG_checkpoint_path ./checkpoints/PIFu_baseline/netG_epoch_44_2415

## 7. Evaluation Metrics

Benchmarks. To evaluate global topology accuracy of meshes, we report Chamfer Distance (x 10000) and Point-to-Surface Distance (x 10000) between the reconstructed human mesh and the ground truth mesh. We also compute Cosine and *L2* distances for the input view normals to measure fine-scale surface details, such as clothes wrinkles. Small values indicate good performance.
|          |   CD  |  PSD  | Cosine |   *L2*   |
|:--------:|:-----:|:-----:|:------:|:------:|
| DeepHuman     | 11.928 | 11.246 | 0.2088 | 0.4647 |
| PIFu     | 2.604 | 4.026 | 0.0914 | 0.3009 |
| Geo-PIFu | 1.742 | **1.922** | 0.0682 | 0.2603 |
| TransPIFu | **1.626** | 2.361 | **0.0514** | **0.2067** |

Please first activate the `opendrEnv` conda environment, because we need to render normal maps using `opendr`.

	conda activate opendrEnv
	cd TransPIFu/

The scripts below are for `TransPIFu`. You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple evaluation scripts in parallel. For example, with 28 splits the whole inference process will take about 1 hrs.

    CUDA_VISIBLE_DEVICES=0 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --splitNum 28 --splitIdx 0
    CUDA_VISIBLE_DEVICES=1 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --splitNum 28 --splitIdx 1
    ... # until --splitIdx 27

The scripts below are for `Geo-PIFu`.

    CUDA_VISIBLE_DEVICES=0 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --splitNum 28 --splitIdx 0
    CUDA_VISIBLE_DEVICES=1 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/GeoPIFu_query --splitNum 28 --splitIdx 1
    ... # until --splitIdx 27

The scripts below are for `PIFu`.

    CUDA_VISIBLE_DEVICES=0 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/PIFu_baseline --splitNum 28 --splitIdx 0
    CUDA_VISIBLE_DEVICES=1 python main_eval_prepare_iccv.py --compute_vn --datasetDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resultsDir ${PREFERRED_DATA_FOLDER}/data/humanRender/transpifuResults/PIFu_baseline --splitNum 28 --splitIdx 1    
    ... # until --splitIdx 27


## 8. Acknowledgements

This repository is built on: [DeepHuman](https://github.com/ZhengZerong/DeepHuman), [PIFu](https://github.com/shunsukesaito/PIFu) and [Geo-PIFu](https://github.com/shunsukesaito/PIFu). Thank the authors for sharing their code!

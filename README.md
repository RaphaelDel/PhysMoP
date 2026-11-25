# PhysMoP: Incorporating Physics Principles for Precise Human Motion Prediction

**Incorporating Physics Principles for Precise Human Motion Prediction** <br />
  [Yufei Zhang](https://zhangy76.github.io/), Jeffrey O. Kephart, Qiang Ji <br /> 
  WACV2024, [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Incorporating_Physics_Principles_for_Precise_Human_Motion_Prediction_WACV_2024_paper.pdf) | [SupMat](https://openaccess.thecvf.com/content/WACV2024/supplemental/Zhang_Incorporating_Physics_Principles_WACV_2024_supplemental.pdf) <br />
![](method.png)

This repository includes the training and evaluation code for the above paper. 

## Installation
### Environment
```bash
conda create -n physmop python=3.8
conda activate physmop
pip install -r requirements.txt
```

### 3D Human Model 
Download the original neutral SMPL model and run the author's `smpl_preprocessing.py` to create the processed pickle with the joint regressors. For SMPL-H, sign in to the MANO downloads page and choose **Extended SMPL+H model (used in AMASS project)** (`smplh.tar.xz`). Extract the male, female, and neutral `model.npz` files and place the models under the repository root as shown below. [SMPL downloads](https://smpl.is.tue.mpg.de/download.php) | [PhysMoP author's SMPL preparation instructions](https://github.com/zhangy76/KNOWN/issues/4#issuecomment-3697054258) | [MANO downloads](https://mano.is.tue.mpg.de/download.php)
```
|--dataset
|  |--smpl_official
|     |--female
|       |--model.npz
|     |--male
|       |--model.npz
|     |--neutral
|       |--model.npz
|     |-- processed_basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

## PhysMoP Model Training & Evaluation
Please follow the procedure below to conduct the experiments on AMASS.
### Data Preparation
Download the raw 3D motion sequence data from [AMASS](https://amass.is.tue.mpg.de/) (should be similar for [Human3.6M](https://drive.google.com/drive/folders/1ySxiuTCSdUEqbgTcx7bx02uMglPOkKjc?usp=sharing) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)), and store them under the current directory as 
```
|--dataset
|  |--data_raw
|     |--AMASS
|       |--ACCAD
|       |--BioMotionLab_NTroje
|       |--CMU
|       |-- .
|       |-- .
|       |-- .
|  |--data_processed
```
Preprocess the raw data
```
python AMASS.py
```

### Training
PhysMoP is trained in three steps:
#### 1. Train the data-driven model
```
python train_script.py --name data --data True --num_epochs 5 --keypoint_loss_weight_data 1 --pose_loss_weight_data 2
```
#### 2. Train the physics-based model with the checkpoint (DATA_CHECKPOINT) obtained from step 1.
```
python train_script.py --name data_physics --data True --physics True --pretrained_checkpoint DATA_CHECKPOINT --keypoint_loss_weight_physics_gt 1 --pose_loss_weight_physics_gt 2 --num_epochs 2 --keypoint_loss_weight_data 1 --pose_loss_weight_data 2 --resume
```
#### 3. Fix the data-driven and physics-based model and train the fusion model with the checkpoint (PHYSICS_CHECKPOINT) obtained from step 2.

```
python train_script.py --name data_physics_fusion --data True --physics True --fix_weight True --fusion True --pretrained_checkpoint PHYSICS_CHECKPOINT --keypoint_loss_weight_physics_gt 1 --pose_loss_weight_physics_gt 2 --num_epochs 2 --keypoint_loss_weight_data 1 --pose_loss_weight_data 2 --keypoint_loss_weight_fusion 1 --pose_loss_weight_fusion 2 
```
The saved model checkpoint can be found at './logs'.

### Evaluation
To evaluate a model saved at CHECKPOINT_PATH, please run
```
python eval.py --dataset AMASS --checkpoint CHECKPOINT_PATH
```

After code re-organization, we re-produced the results as

|  AMASS-Test  |   80ms |  160ms |  320ms |  400ms |  560ms |  720ms |  880ms | 1000ms |
| ------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  MPJPE (mm)  |    0.4 |    2.0 |    9.4 |   14.2 |   24.4 |   36.1 |   48.0 |   61.9 |

The evaluated checkpoint, which includes model weights and training log, can be downloaded via [FINAL_MODEL](https://www.dropbox.com/scl/fi/qfovsy30net1wt2zl04bw/2023_12_21-17_09_24_20364.pt?rlkey=2phz52d26p2w2hbtx50o89c50&dl=0).

### Human3.6M evaluation

Place the NeuralAnnot SMPL JSON files in `dataset/dataset_raw/H36M`, or pass their directory with `--raw-dir`. From the repository root, run:

```bash
python H36M.py --split test
python eval.py --dataset H36M --checkpoint path/to/checkpoint.pt --batch-size 8
```

The test split uses subject S5, as in the paper. The converter downsamples from 50 to 25 Hz, averages the ten shape coefficients per recording, applies the AMASS smoothing, and writes non-overlapping windows with 25 history frames and 25 future frames. It saves the index at `dataset/data_processed/h36m_test_50.pkl`. The evaluator reports MPJPE at each horizon (P1) and averaged over future frames through that horizon (P2). The choice of non-overlapping windows may affect comparison with Table 1. [Author's preprocessing notes](https://github.com/zhangy76/PhysMoP/issues/5#issuecomment-2211809103) · [PhysMoP paper, Section 4](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Incorporating_Physics_Principles_for_Precise_Human_Motion_Prediction_WACV_2024_paper.pdf#page=5)

Evaluation requires the processed neutral SMPL pickle and the 6,890-vertex male and female SMPL-H archives described above. The loader accepts prepared 24-joint `weights_prior` or the official archive's 52-joint `weights`. For 52-joint archives it combines each hand's finger weights into the corresponding hand joint used by PhysMoP. [SMPL-H joint definitions](https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py#L164)

The linked `FINAL_MODEL` checkpoint is for AMASS and can be used for a cross-dataset comparison. For an in-dataset evaluation, train on Human3.6M with S5 held out as described below.

### Human3.6M training

Run `python H36M.py --split both` once to create the train and test indexes. The training split uses S1, S6, S7, S8, S9, and S11. The test split uses S5.

Train the data-driven branch for 30 epochs, add the physics branch for 10 epochs, then freeze both branches and train fusion for 10 epochs. The paper used batch sizes of 64, 64, and 196. The PowerShell example starts with eight clips per batch and adjusts the decay interval so that each decay comes after roughly the same number of training clips as in the paper. Set the batch sizes to 64 and 196 if your GPU can fit them. [PhysMoP paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Incorporating_Physics_Principles_for_Precise_Human_Motion_Prediction_WACV_2024_paper.pdf#page=5) · [Supplementary material](https://openaccess.thecvf.com/content/WACV2024/supplemental/Zhang_Incorporating_Physics_Principles_WACV_2024_supplemental.pdf)

```powershell
$branchBatch = 8
$fusionBatch = 8
$branchDecay = [int][Math]::Ceiling(500 * 64 / $branchBatch)
$fusionDecay = [int][Math]::Ceiling(500 * 196 / $fusionBatch)
$trainOptions = @('--dataset', 'H36M', '--num_workers', '0', '--lr_gamma', '0.9')

python train_script.py @trainOptions --name h36m_data --data True --num_epochs 30 `
  --batch_size $branchBatch --lr_step_size $branchDecay `
  --keypoint_loss_weight_data 1 --pose_loss_weight_data 2
if ($LASTEXITCODE -ne 0) { throw 'Data-driven training failed' }
$dataCheckpoint = (Get-ChildItem logs/h36m_data/checkpoints -Filter *.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

python train_script.py @trainOptions --name h36m_physics --data True --physics True --num_epochs 10 `
  --batch_size $branchBatch --lr_step_size $branchDecay --pretrained_checkpoint $dataCheckpoint `
  --keypoint_loss_weight_data 1 --pose_loss_weight_data 2 `
  --keypoint_loss_weight_physics_gt 1 --pose_loss_weight_physics_gt 2
if ($LASTEXITCODE -ne 0) { throw 'Physics training failed' }
$physicsCheckpoint = (Get-ChildItem logs/h36m_physics/checkpoints -Filter *.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

python train_script.py @trainOptions --name h36m_fusion --data True --physics True --fusion True --fix_weight True --num_epochs 10 `
  --batch_size $fusionBatch --lr_step_size $fusionDecay --pretrained_checkpoint $physicsCheckpoint `
  --keypoint_loss_weight_fusion 1 --pose_loss_weight_fusion 2 --fusion_weight_reg_loss_weight 1
if ($LASTEXITCODE -ne 0) { throw 'Fusion training failed' }
$fusionCheckpoint = (Get-ChildItem logs/h36m_fusion/checkpoints -Filter *.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

python eval.py --dataset H36M --checkpoint $fusionCheckpoint --batch-size 8
```

The example evaluates all non-overlapping S5 windows produced by `H36M.py`. The paper evaluated 256 sampled windows using the siMLPe protocol. [Supplementary material](https://openaccess.thecvf.com/content/WACV2024/supplemental/Zhang_Incorporating_Physics_Principles_WACV_2024_supplemental.pdf)

## Citation
If you find our work useful, please consider citing the paper:
```bibtex
@InProceedings{Zhang_2024_WACV,
    author    = {Zhang, Yufei and Kephart, Jeffrey O. and Ji, Qiang},
    title     = {Incorporating Physics Principles for Precise Human Motion Prediction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {6164-6174}
}
```

If you have questions or encouter any issues when running the code, feel free to open an issue or directly contact me via: zhangy76@rpi.edu.

## References
Part of our code is developed based on [siMLPe](https://github.com/dulucas/siMLPe). The SMPL model data is downloaded from [SMPL-X model](https://github.com/vchoutas/smplx). We thank them for generously sharing their outstanding work.

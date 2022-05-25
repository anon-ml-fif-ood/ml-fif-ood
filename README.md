# A Functional Perspective for Multi-Layer Out-of-Distribution Detection

![Summary of how our method works.](images/main_diagram.png)

Anonymized Code for the paper "A Functional Perspective for Multi-Layer Out-of-Distribution Detection".

Submitted at NeurIPS 2022.

Please check the references for more details on how to use the resources.

## Specification of dependencies

You should have the following dependencies pre-installed:

```bash
python>=3.8
torch>=1.10
torchvision>=0.11
```

Then, install the remaining dependencies with `pip install -r requirements.txt`.

Optional: using a virtual environment:

```bash
python -m venv env 
source env/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

## Environmental Variables (Optional):

Set the environment variables from `.env.example` file in a `.env` file.

## Pre-trained models

Download pre-trained models and place them in the `PRE_TRAINED_DIR` environmental variable (e.g., `PRE_TRAINED_DIR="${PWD}/vision/pre_trained"`).

```bash
# BiT-S-R101
wget -P $PRE_TRAINED_DIR https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz 
# ViT-L-16
wget -P $PRE_TRAINED_DIR https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz
```

Or simply run the script `bash download.sh`.

## Datasets

### In-distribution: ImageNet

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the data in the directory set by the environmental variable `IMAGENET_ROOT` (e.g., `IMAGENET_ROOT="data/ILSVRC2012/"`). 



### Out-of-distribution

The OOD datasets will be download automatically when called (see `vision/datasets` files). We considered the same datasets as `https://github.com/deeplearning-wisc/large_scale_ood/`. 

Alternatively, you can download them manually by running the script and placing them in the directory set by the environmental variable `DATASETS_DIR` (e.g., `DATASETS_DIR="data/"`).

```bash
wget -P $DATASETS_DIR https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
wget -P $DATASETS_DIR http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget -P $DATASETS_DIR http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget -P $DATASETS_DIR http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

## Our Algorithm

We implement our Class Conditional Functional Isolation Forest algorithm in the file `aggregations/ccfif.py`. The FIF package was automatically installed from `https://github.com/GuillaumeStaermanML/FIF`.

## Evaluation: Reproducing our Results and Baselines

In order to reproduce the main results of our paper, please run the following scripts:

```shell
bash scripts/0_save_features.sh
bash scripts/1_save_functional_dataset.sh
bash scripts/3_results.sh
bash scripts/4_baselines.sh
```

These scripts will:

1. Save the latent features of the three models in memory to the directory `tensors/` (approx 207GB).
2. Map the features into a functional dataset and save it to the directory `tensors/`.
3. Calculate the FIF OOD scores and the evaluation metrics and save the results to `results/`.
4. Calculate the baseline scores and the evaluation metrics and save the results to `results/`.

## Results

We report our main results in Table~\ref{tab:main} which includes the evaluation metrics for the three pre-trained architectures, four OOD datasets, and six detection methods.

Table 1. Comparing our method against state-of-the-art on OOD detection. Our scores are averaged over 10 seeds. Values are in percentage.

| **Models** | **Methods** | **iNaturalist**| | **SUN** | |**Places** | | **Textures** | |
|---|---|---|---|---|---|---|---|---|---|
|  |  | TNR95 | AUROC | TNR95 | AUROC | TNR95 | AUROC | TNR95 | AUROC |
| BiT-S-101 | MSP | 36.31 | 87.59 | 20.02 | 78.34 | 18.56 | 76.76 | 17.27 | 74.45 |
|  | ODIN | 37.31 | 89.36 | 28.33 | 83.92 | 23.73 | 80.67 | 18.69 | 76.30 |
|  | Energy | 35.09 | 88.48 | 34.67 | 85.32 | 26.98 | 81.37 | 19.13 | 75.79 |
|  | Maha. | 3.66 | 46.33 | 11.57 | 65.20 | 10.25 | 64.46 | 47.77 | 72.10 |
|  | GradN. | 49.97 | 90.33 | 53.52 | 89.03 | 39.14 | 84.82 | 38.58 | 81.07 |
|  | Ours | 100 | 98.92 | 100 | 99.10 | 100 | 99.08 | 100 | 98.90 |
| DenseNet-121 | MSP | 51.45 | 89.16 | 30.61 | 80.46 | 28.58 | 80.11 | 31.49 | 78.69 |
|  | ODIN | 63.00 | 93.29 | 42.7 | 86.12 | 38.09 | 84.14 | 43.51 | 84.62 |
|  | Energy | 63.61 | 93.29 | 45.09 | 86.53 | 40.02 | 84.29 | 46.13 | 85.07 |
|  | Maha. | 2.64 | 42.24 | 1.76 | 41.17 | 2.68 | 47.27 | 37.22 | 56.53 |
|  | GradN. | 76.13 | 93.97 | 56.96 | 87.79 | 46.08 | 83.04 | 56.84 | 87.48 |
|  | Ours | 99.97 | 98.60 | 100 | 98.71 | 100 | 98.70 | 100 | 98.23 |
| ViT-16-L | MSP | 81.29 | 95.89 | 47.38 | 87.51 | 43.43 | 85.90 | 47.55 | 85.08 |
|  | ODIN | 90.36 | 98.15 | 62.13 | 93.09 | 54.41 | 90.63 | 59.72 | 90.81 |
|  | Energy | 91.53 | 98.35 | 65.43 | 93.61 | 56.73 | 91.00 | 62.22 | 91.30 |
|  | Maha. | 97.88 | 99.52 | 65.00 | 93.40 | 54.81 | 90.38 | 60.16 | 90.61 |
|  | GradN. | 100 | 100 | 99.90 | 99.90 | 85.90 | 97.60 | 86.30 | 96.80 |
|  | Ours | 100 | 99.98 | 100 | 99.99 | 100 | 99.98 | 100 | 99.99 |

#### References

* [Pytorch](https://pytorch.org/)
* [FIF](https://github.com/GuillaumeStaermanML/FIF)
* [BiT pre-trained models](https://github.com/google-research/big_transfer)
* [ViT pre-trained models](https://github.com/google-research/vision_transformer)
* [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index)
* [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space](https://github.com/deeplearning-wisc/large_scale_ood/)

<!-- ## Cite this work

 -->

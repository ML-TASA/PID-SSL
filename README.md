# PID-SSL

![Static Badge](https://img.shields.io/badge/ICML25-yellow)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)

Pytorch implementation of [ICML2025] On the Out-of-Distribution Generalization of Self-Supervised Learning

This repository implements classic SSL methods such as **SimCLR**, **MoCo**, and **BYOL**, integrated with **PIB Sampling**. The setup is designed to be **plug-and-play**, allowing you to seamlessly switch between different SSL models and datasets without additional configuration. Simply configure the arguments, and you're ready to train!

The provided command-line examples are ready-to-use, allowing you to get started with training your models right away.


## Data Preparation

The data should be organized in the following directory structure:

```
./data/mltasa/
    |--- CIFAR10/
    |--- Image-100/
    |--- ImageNet/
```

### Example Datasets:

* **CIFAR-10**: Downloadable via `torchvision`, or place it in the `CIFAR10` directory.
* **Image-100**: A custom dataset with the usual folder structure (one folder per class). Ensure it¡¯s structured correctly within the `Image-100` directory.
* **ImageNet**: Available for download from [ImageNet's official site](http://www.image-net.org/download-images). Ensure it's placed under the `ImageNet` folder.



## Installation

To set up your environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository/ssl-pib-sampling.git
   cd ssl-pib-sampling
   ```

2. Install the required Python dependencies:

```
torch==1.10.0
torchvision==0.11.1
PILLOW==8.3.1
numpy==1.21.2
argparse==1.4.0
```

---

## Training

Once your data is prepared and dependencies are installed, you're ready to train an SSL model using PIB sampling.

### Run Example

Our approach is **plug-and-play**, meaning the following examples are ready to use. You only need to configure the arguments for your preferred dataset and model type.

* **SimCLR on CIFAR-10**:

  ```bash
  python main_pib.py --batch_size 128 --epochs 100 --learning_rate 0.001 --model_type SimCLR --dataset_name CIFAR10 --data_dir ./data/mltasa
  ```

* **MoCo on Image-100**:

  ```bash
  python main_pib.py --batch_size 64 --epochs 50 --learning_rate 0.0005 --model_type MoCo --dataset_name Image-100 --data_dir ./data/mltasa
  ```

* **BYOL on ImageNet**:

  ```bash
  python main_pib.py --batch_size 256 --epochs 200 --learning_rate 0.0001 --model_type BYOL --dataset_name ImageNet --data_dir ./data/mltasa
  ```


## Cite

```
@misc{qiang2025outofdistributiongeneralizationselfsupervisedlearning,
      title={On the Out-of-Distribution Generalization of Self-Supervised Learning}, 
      author={Wenwen Qiang and Jingyao Wang and Zeen Song and Jiangmeng Li and Changwen Zheng},
      year={2025},
      eprint={2505.16675},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.16675}, 
}
```


### Notes:

* The setup is **plug-and-play**, meaning you can easily switch between datasets and SSL methods with minimal configuration.
* Simply adjust the `--model_type` and `--dataset_name` parameters to switch models and datasets.
* The code includes PIB sampling by default, ensuring high-quality and diverse mini-batches during training.

---

# SuSy: Synthetic Image Detection and Attribution

SuSy is a Spatial-Based Synthetic Image Detection and Recognition Model, designed to detect synthetic images and attribute them to specific generative models. This repository provides the code and instructions to train and evaluate SuSy or your own model for synthetic image detection.

## Quick Links
- **Model:** [HPAI-BSC/SuSy on Hugging Face](https://huggingface.co/HPAI-BSC/SuSy)
- **Dataset:** [HPAI-BSC/SuSy-Dataset on Hugging Face](https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset)
- **Model Demo:** https://colab.research.google.com/drive/15nxo0FVd-snOnj9TcX737fFH0j3SmS05

## Table of Contents
- [Installation](#installation)
- [Model Overview](#model-overview)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Authors](#authors)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/HPAI-BSC/SuSy.git
   ```
2. Navigate to the project directory:
   ```
   cd SuSy
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Model Overview

SuSy is a CNN-based model that takes image patches of size 224x224 as input and outputs probabilities for the image being authentic or generated by specific AI models (e.g., StableDiffusion, Midjourney, DALL·E 3). Key features:

- Architecture: CNN feature extractor (ResNet-18 based) + Multi-Layer Perceptron
- Total parameters: 12.7M
- Input: 224x224 image patches
- Output: Probabilities for 6 classes (5 synthetic models + 1 real image class)

For detailed architecture information, see the [model card](https://huggingface.co/HPAI-BSC/SuSy).

## Usage

1. Configure paths in `staircase_fe.yaml` or a custom experiment YAML file.
2. Train the model:
   ```
   python3 src/train.py experiment=staircase_fe
   ```
3. Evaluate the model:
   ```
   python3 src/eval.py experiment=staircase_fe
   ```

## Dataset

The SuSy Dataset comprises authentic and synthetic images from various sources:

|      Dataset      | Year | Train | Validation |  Test | Total |
|:-----------------:|:----:|:-----:|:----------:|:-----:|:-----:|
|        COCO       | 2017 | 2,967 |    1,234   | 1,234 | 5,435 |
|   dalle-3-images  | 2023 |  987  |     330    |  330  | 1,647 |
|    diffusiondb    | 2022 | 2,967 |    1,234   | 1,234 | 5,435 |
|   realisticSDXL   | 2023 | 2,967 |    1,234   | 1,234 | 5,435 |
|   midjourney-tti  | 2022 | 2,718 |     906    |  906  | 4,530 |
| midjourney-images | 2023 | 1,845 |     617    |  617  | 3,079 |

For more details on the dataset composition and licensing, refer to the [dataset card](https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset).

## Training

Key training details:

- Patch extraction: 240x240 patches, selected based on gray-level co-occurrence matrix (GLCM) contrast
- Data augmentation: Includes flipping, brightness/contrast adjustment, blurring, and JPEG compression
- Optimizer: Adam with learning rate 0.0001
- Loss function: Cross-Entropy Loss
- Batch size: 128
- Epochs: 10 (with early stopping)

For full training procedure and hyperparameters, see the [model card](https://huggingface.co/HPAI-BSC/SuSy).

## Evaluation

SuSy has been evaluated on various datasets, including:

- Test split of the training dataset
- Synthetic images from newer models (e.g., Stable Diffusion 3, FLUX.1-dev)
- In-the-wild synthetic and authentic images
- Flickr 30k, Google Landmarks v2, and Synthbuster datasets

Evaluation metrics focus on recall. For detailed results, see the [model card](https://huggingface.co/HPAI-BSC/SuSy).

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make changes and commit: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Report issues or suggestions in the [issue tracker](https://github.com/HPAI-BSC/SuSy/issues).

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Citation

```bibtex
@misc{bernabeu2024susy,
    title={Present and Future Generalization of Synthetic Image Detectors},
    author={Bernabeu Perez, Pablo and Lopez Cuena, Enrique and Garcia Gasulla, Dario},
    year={2024},
    month={09}
}

@thesis{bernabeu2024aidetection,
    title={Detecting and Attributing AI-Generated Images with Machine Learning},
    author={Bernabeu Perez, Pablo},
    school={UPC, Facultat d'Informàtica de Barcelona, Departament de Ciències de la Computació},
    year={2024},
    month={06}
}
```

## Authors

This repository was created by [Pablo Bernabeu Perez](https://github.com/pbernabeup) and [Enrique Lopez Cuena](https://huggingface.co/Cuena).

For inquiries, contact [HPAI](mailto:hpai@bsc.es).
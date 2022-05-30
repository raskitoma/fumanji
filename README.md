# Fumanji

Applying a random picture style to a set of images preserving original image size.
## Code Quality

[![Quality gate](https://sonarqube.raskitoma.com/api/project_badges/quality_gate?project=fumanji&token=8f5b74a0a7e0d8c4ee52972856268fa27bb8d5ed)](https://sonarqube.raskitoma.com/dashboard?id=fumanji)
## Config

Clone this repo and cd to the directory.

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Drop your content images into the **"img"** folder.

Drop image styles into the **"styles"** folder.

Run the script using the following command:

```bash
python fumanji.py
```

> The script will generate a set of images based on the content images and applying a random image style.
> The generated images will be saved in the **"results"** folder.

## Special info

> Take note that each time the script runs it will empty the **"results"** folder.

## How to speed up process

To speed up, if you have a NVIDIA GPU, you must install the following stuff:

1. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) - Please, check before download, so you can download the one that suits your environment.
2. [NVIDIA cuDNN(NVIDIA CUDA Neural Network Library)](https://developer.nvidia.com/cudnn) - Download, unpack and copy contents from **bin** to **NVIDIA CUDA Toolkit** folder (example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`)

To download both CUDA and cuDNN, you need to have a NVIDIA Developer account.

> If you found some error at run, like Error 193, check if your CUDA version is compatible with cuDNN.  You must download/install compatible CUDA and cuDNN. Also check if your Tensorflow is compatible with your CUDA version.

## Improvements

It is supposed that you can improve this script using CUDA.

## Credits

Thanks to [Behic Guven article](https://towardsdatascience.com/python-for-art-fast-neural-style-transfer-using-tensorflow-2-d5e7662061be).

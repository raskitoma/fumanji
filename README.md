# Fumanji

Applying a random picture style to a set of images preserving original image size using Tensorflow.
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

## Parameters

```bash
usage: fumanji.py [-h] [-o OUTPUT_SIZE] [-s STYLE_SIZE] [-t] [-q QUANTITY] [-i ITERATIONS] [-f INTENSITY] [-r]

options:
  -h, --help            show this help message and exit
  -o OUTPUT_SIZE, --output_size OUTPUT_SIZE
                        Image output size. [small, medium, large, original] GPU memory can prevent using 'original' image size.
                        (default: medium)
  -s STYLE_SIZE, --style_size STYLE_SIZE
                        Style size. [xsmall, small, medium, large, xlarge] (default: small)
  -t, --trained         Uses pre-trained model. (default: False)
  -q QUANTITY, --quantity QUANTITY
                        How many images to process. (default: None)
  -i ITERATIONS, --iterations ITERATIONS
                        How many iterations to apply. Min 1, Max 100. (default: 1)
  -f INTENSITY, --intensity INTENSITY
                        Effect Strenght. (default: 30)
  -r, --show_result     Shows result after each iteration. (default: False)
```

## Special info

> Take note that each time the script runs it will empty the **"results"** folder.

## How to speed up process

> Tested on Windows

To speed up, if you have a NVIDIA GPU, you must install the following stuff:

1. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) - Please, check before download, so you can download the one that suits your environment.
2. [NVIDIA cuDNN(NVIDIA CUDA Neural Network Library)](https://developer.nvidia.com/cudnn) - Download, unpack and copy contents from **cuda** to **NVIDIA CUDA Toolkit** folder. (example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`)

**To download both CUDA and cuDNN, you need to have a NVIDIA Developer account.**

> If you found some error at run, like Error 193, check if your CUDA version is compatible with cuDNN.  You must download/install compatible CUDA and cuDNN. Also check if your Tensorflow is compatible with your CUDA version.  Documentation suggest to use CUDA Toolkit 11.2 and cuDNN 8.1.0.

## Speed up with Apple M1

For this, you need to install Tensorflow for Mac.  The process is:

1. Install Anaconda for Mac
2. Install [Miniforge](https://github.com/conda-forge/miniforge): This allows to download packages precompiled for Apple Silicon (arm). Running: `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh` or
`wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh` should do the trick.
3. Follow this article by *Prabat* with all the steps [here](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)

## Improvements

It is supposed that you can improve this script using CUDA.

## Credits

Thanks to [Behic Guven article](https://towardsdatascience.com/python-for-art-fast-neural-style-transfer-using-tensorflow-2-d5e7662061be).  It was a great help to understand how to use Tensorflow.

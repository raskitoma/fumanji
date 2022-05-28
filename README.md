# Fumaji

Applying a random picture style to a set of images preserving original image size.

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
fumanji.py
```

> The script will generate a set of images based on the content images and applying a random image style.
> The generated images will be saved in the **"results"** folder.
> {.is-info}

## Special info

> Take note that each time the script runs it will empty the **"results"** folder.
> {.is-warning}

## Improvements

It is supposed that you can improve this script using CUDA.

## Credits

Thanks to [Behic Guven article](https://towardsdatascience.com/python-for-art-fast-neural-style-transfer-using-tensorflow-2-d5e7662061be).

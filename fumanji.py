from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import datetime
import random

# function to load an image from a path using original size (no resize) any format
# def load_image(path):
#     img = tf.io.read_file(path), 
#     img = tf.image.decode_image(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     return img

# def load_image(image_path, image_size=(2048, 2048)):
#     img = tf.io.decode_image(
#       tf.io.read_file(image_path),
#       channels=3, dtype=tf.float32)[tf.newaxis, ...]
#     img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
#     return img

def load_image(image_path):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    return img


def visualize(images, titles=('',)):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i])
        plt.savefig("final.jpg")
    plt.show()

def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)


# Function to load all images jpg or png on img folder
def load_images(path):
    images = []
    for img_path in tf.io.gfile.glob(path):
        img = load_image(img_path)
        images.append(img)
    return images

# load random image from img folder 
def load_random_image(path):
    img_path = random.choice(tf.io.gfile.glob(path))
    img = load_image(img_path)
    return img

# load all images from /img folder
images = load_images('img/*')

# delete all files from /results folder
for file in tf.io.gfile.glob('results/*'):
    tf.io.gfile.remove(file)

# for each image, run the model and visualize the result using images and style_image
for i in range(len(images)):

    # print image name
    print( "Image: " + str(i) )

    # set start time for this iteration
    begin_time = datetime.datetime.now()

    # load image
    original_image = images[i]

    # load one random image from /style folder
    style_image = load_random_image('styles/*')

    # rezise style_image to 256x256 to improve visual results
    style_image = tf.image.resize(style_image, [256, 256])

    # run the model       
    # style_image = tf.nn.avg_pool(style_image, [1, 4, 4, 1], [1, 4, 4, 1], 'SAME')
    # style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    stylize_model = tf_hub.load('tf_model')
    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
    stylized_photo = results[0]

    # set a unique name for the image in the /results folder
    img_name = "results/result_" + str(i) + ".png"
    export_image(stylized_photo).save(img_name)

    # print execution time
    print("Execution time for: " + str(datetime.datetime.now() - begin_time))

    # visualize the result
    visualize([original_image, style_image, stylized_photo], titles=['Original Image', 'Style Image', 'Result Image'])

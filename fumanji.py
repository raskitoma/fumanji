# Fumanji.py
# Fumanji.py
# Image style transfer using Tensorflow
# (c) 2022, Raskitoma, https://raskitoma.com

# from curses import flushinp
from turtle import up
from matplotlib import gridspec, image
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import datetime
import random
import os
import argparse
import functools

# reading parameters
parser = argparse.ArgumentParser(
                                 description='Fumanji - Image style transfer using Tensorflow',
                                 epilog='(c) 2022, Raskitoma, https://raskitoma.com',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 allow_abbrev=True,
                                 exit_on_error=True
                                )
parser.add_argument("-o", "--output_size", help="Image output size. [small, medium, large, original]\nGPU memory can prevent using 'original' image size.", default='medium')
parser.add_argument("-s", "--style_size", help="Style size. [xsmall, small, medium, large, xlarge]", default='small')
parser.add_argument("-t", "--trained", help="Uses pre-trained model.", action="store_true")
parser.add_argument("-q", "--quantity", help="How many images to process.", default=None)
parser.add_argument("-i", "--iterations", help="How many iterations to apply. Min 1, Max 100.", default=1)
parser.add_argument("-f", "--intensity", help="Effect Strenght.", default=30)
parser.add_argument("-r", "--show_result", help="Shows result after each iteration.", action="store_true")
args = parser.parse_args()

# read style size parsed
def read_style_size(style_size):
    if style_size == 'xsmall':
        return [128, 128]
    elif style_size == 'small':
        return [256, 256]
    elif style_size == 'medium':
        return [512, 512]
    elif style_size == 'large':
        return [1024, 1024]
    elif style_size == 'xlarge':
        return [2048, 2048]
    else:
        print('Style size not recognized. Accepted values are xsmall, small, medium, large and xlarge')
        exit()

# read output size parsed
def read_output_size(output_size):
    if output_size == 'small':
        return [256, 256]
    elif output_size == 'medium':
        return [512, 512]
    elif output_size == 'large':
        return [1024, 1024]
    elif output_size == 'original':
        return None
    else:
        print('Output size not recognized. Accepted values are small, medium, large and original')
        exit()

# set style size based on args.style_size if not set, default is small
style_size = read_style_size(args.style_size)

# set output size based on args.output_size if not set, default is medium
output_size = read_output_size(args.output_size)

# set show_result base on args if not set, default is False
show_result = args.show_result

# set use of trained base on args if not set, default is False
use_trained = args.trained

# set quantity of images to process
if args.quantity:
    quantity = int(args.quantity)
else:
    quantity = len(tf.io.gfile.glob('./img/*'))

# set quantity of iterations
if args.iterations:
    iterations = int(args.iterations)
else:
    iterations = 0

if iterations > 100 or iterations < 1:
    print('Iterations must be between 1 and 100')
    exit()

# set intensity of effect
total_variation_weight = int(args.intensity)

## Setting parameters
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

## Setting functions

# Function to load image
def load_image(image_path):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, :]
    return img

# Function to visualize images
def visualize(images, titles=('',)):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] + image_sizes[1]) // 100
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i])
    plt.show()

# Function to export image to png
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
        # skip if not a jpg or png
        if not img_path.endswith('.jpg') and not img_path.endswith('.jpeg') and not img_path.endswith('.png'):
            continue
        img = load_image(img_path)
        images.append(img)
    return images

# load random image from img folder 
def load_random_image(path):
    options = tf.io.gfile.glob(path)
    # Choose random image and repeat if file is not an image
    while True:
        img_path = random.choice(options)
        if not img_path.endswith('.jpg') and not img_path.endswith('.jpeg') and not img_path.endswith('.png'):
            continue
        img = load_image(img_path)
        return img

# Functions for on demand training
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))  

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

###############################################################################
# Start

# clearing screen
if(os.name == 'posix'):
    os.system('clear')
else:
    os.system('cls')

# Prepare tensorflow model load
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

print("Fumanji - Image Style Transfer by Raskitoma.com - Version 1.0\n")

print("TF Version: ", tf.__version__)
print("TF Hub version: ", tf_hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# if GPU available, full GPU data

if use_trained:
    print("*** Training disabled. Using trained model.")
else:
    print("*** Training enabled")
    print("Training step iterations: ", iterations)
    print("Intensity: ", total_variation_weight)

if quantity: 
    print("Process only ", quantity, " images")
else:
    print("Process all images")

if show_result:
    print("Showing results after each processed image")

master_start_time = datetime.datetime.now()

print("==========================================================")
print("Start: " + str(master_start_time))
print("==========================================================")

# delete all files from /results folder except .gitignore
print("===== Clearing results folder...")
for file in tf.io.gfile.glob('results/*'):
    #skip if filename is .gitignore
    if file == 'results/.gitignore':
        continue
    tf.io.gfile.remove(file)


# load all images from /img folder
print("===== Loading images into memory array...")
images = load_images('img/*')

# Load model
print("===== Loading required model...")
if use_trained:
     stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# else:
#     vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#     print()
#     for layer in vgg.layers:
#         print(layer.name)

print("===== Model loaded.")

# print separator
print("==========================================================")

# for each image, run the model and visualize the result using images and style_image
print("===== Stylizing images...")
for i in range(len(images)):

    # if quantity is 0, loop continues. If quantity is > 0 and i+1 is surpassed, loop ends
    if quantity > 0 and i+1 > quantity:
        break      

    # set start time for this iteration
    iteration_begin_time = datetime.datetime.now()

    # print image number
    print( "Procesing Image " + str(i+1) + " of " + str(len(images)))

    # load one image from img array
    original_image = images[i]
    if output_size:
        original_image = tf.image.resize(original_image, output_size)
    else:
        print("No output size specified. Using original image size.")
        print("For original size is recommended to use the trained model.")
        print("Other option is using a GPU with lots of free memory.")

    # load one random image from /style folder
    style_image = load_random_image('styles/*')
    # rezise style_image to 256x256. Required for model input and handling
    style_image = tf.image.resize(style_image, style_size)
    # setting the style model 
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    #  checks if trained mode is used
    if use_trained:
        print("===== Using trained model...")

        # applying model to image
        results = stylize_model(tf.constant(original_image), tf.constant(style_image))

        # grabbing the image from the results
        stylized_photo = results[0]
    
    else:
        print("===== Using std model...")
        # x= tf.keras.applications.vgg19.preprocess_input(original_image)

        content_layers = ['block5_conv2'] 
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        style_extractor = vgg_layers(style_layers)
        style_outputs = style_extractor(style_image*255)
        #Look at the statistics of each layer's output
        for name, output in zip(style_layers, style_outputs):
            print(name)
            print("  shape: ", output.numpy().shape)
            print("  min: ", output.numpy().min())
            print("  max: ", output.numpy().max())
            print("  mean: ", output.numpy().mean())
            print()        
        extractor = StyleContentModel(style_layers, content_layers)
        results = extractor(tf.constant(original_image))
        print('Styles:')
        for name, output in sorted(results['style'].items()):
            print("  ", name)
            print("    shape: ", output.numpy().shape)
            print("    min: ", output.numpy().min())
            print("    max: ", output.numpy().max())
            print("    mean: ", output.numpy().mean())
            print()
        print("Contents:")
        for name, output in sorted(results['content'].items()):
            print("  ", name)
            print("    shape: ", output.numpy().shape)
            print("    min: ", output.numpy().min())
            print("    max: ", output.numpy().max())
            print("    mean: ", output.numpy().mean())        
        style_targets = extractor(style_image)['style']
        content_targets = extractor(original_image)['content']

        image = tf.Variable(original_image)

        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        style_weight = 1e-2
        content_weight = 1e4

        tf.image.total_variation(image).numpy()        

        print("===== Starting training...")
        steps_per_epoch = 100
        step = 0
        for n in range(iterations):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print(".", end='', flush=True)
                if step % 10 == 0:
                    print("{}%".format(step), end='', flush=True)
            print(" - Train step: {}".format(n+1))
        stylized_photo = image        
    
    # set a unique name for the image in the /results folder and save it
    img_name = "results/result_" + str(i) + ".png"
    export_image(stylized_photo).save(img_name)

    # print execution time
    print("Execution time: " + str(datetime.datetime.now() - iteration_begin_time))

    # show result
    if show_result:
        visualize([original_image, style_image, stylized_photo], ['Original', 'Style', 'Stylized'])   

master_end_time = datetime.datetime.now()
print("==========================================================")
print("End: " + str(master_end_time))
print("Total time: " + str(master_end_time - master_start_time))
print("==========================================================")
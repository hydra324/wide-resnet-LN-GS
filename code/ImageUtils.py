import numpy as np
import random
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    image = record.reshape((3, 32, 32))
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(image, [1, 2, 0])  # new shape: [32,32,3]

    if training:
        # Resize the image to add four extra pixels on each side.
        image = np.pad(image,((2,2),(2,2),(0,0)),'constant')

        # Randomly crop a [32, 32] section of the image.
        [top_left_x,top_left_y] = np.random.randint(low=0,high=4+1,size=(2,))
        image = image[top_left_x:top_left_x+32,top_left_y:top_left_y+32,:]

        # Randomly flip the image horizontally.
        if random.random() > 0.5:
            image = np.flip(image,axis=1)

    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image,axis=(1,2),keepdims=True)
    std = np.std(image,axis=(1,2),keepdims=True)
    image = (image-mean)/(std+1e-5)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1]) # new shape: [3,32,32]

    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image = image.reshape((3,32,32))
    image = np.transpose(image, [1,2,0])
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE
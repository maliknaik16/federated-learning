
"""
This python script preprocesses the data from the JSON file into a CSV file
that can be easily loaded using the Pandas.

Author: Malik Naik
Website: http://maliknaik.me/
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os

# Set the data path.
path = 'SET THE PATH TO THE TRAIN DIRECTORY HERE' # './leaf/data/femnist/data/train'

if __name__ == '__main__':

    # Initialize the list to store all the images and their labels.
    fmnist = []

    for file in os.listdir(path):
        if file.endswith('.json'):
            # Open the json file with the image data.
            f = open(file)

            # Load the json file.
            data = json.load(f)

            # for each user.
            for _, v in data['user_data'].items():

                # for each row/record.
                for i in range(len(v['y'])):
                    # Get the image as list.
                    img = np.array(v['x'][i]).reshape(-1, ).tolist()

                    # Append the label to the list.
                    img.append(v['y'][i])

                    # Append the final image and it's label.
                    fmnist.append(img)

    with open('FEMNIST.csv', 'a') as f:

        for row in fmnist:
            x = ','.join(map(str, row)) + "\n"
            f.write(x)

        f.close()

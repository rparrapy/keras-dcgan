"""
Utility functions to load the Cohn-Kanade dataset.
"""
import fnmatch
import os
from scipy import misc
import numpy as np

def load_data(path="/mnt/raid/data/ni/dnn/ck+/"):
    """Load training and test sets from the Cohn-Kanade dataset.

    The path directory should contain:
    - A folder named 'Emotion' with the unzipped content of 'Emotion_labels.zip'
    - A folder named 'cohn-kanade-images' with the unzipped content of 'Emotion_labels.zip'

    :param path: the path to folder where the dataset is downloaded
    :returns: ((X_train, y_train), (X_test, y_test))
    """

    #matches = [path + 'Emotion/S011/001/S011_001_00000016_emotion.txt']
    matches = []
    for root, _, filenames in os.walk(path + 'Emotion'):
        for filename in fnmatch.filter(filenames, '*.txt'):
            matches.append(os.path.join(root, filename))

    to_image = lambda x: x.replace('Emotion', 'cohn-kanade-images').replace('_emotion.txt', '.png')
    images = []
    emotions = []
    for match in matches:
        emotion_image = to_image(match)
        neutral_image = '_'.join(emotion_image.split('_')[:-1]) + '_00000001.png'
        resized_emotion = misc.imresize(misc.imread(emotion_image), (64, 64))
        resized_neutral = misc.imresize(misc.imread(neutral_image), (64, 64))
        images.append(resized_emotion)
        images.append(resized_neutral)
        with open(match, 'r') as emotion_file:
            emotion = int(float(emotion_file.readline()))
            emotions.append(emotion)
        emotions.append(0)

    return ((np.array(images), None), (None, None))
    # for i, image in enumerate(images):
    #     print image
    #     print emotions[i]
    # print 'here!'


def main():
    """
    Main function for local testing.
    """
    load_data('/Users/rparra/Workspace/tub/thesis/data/')

if __name__ == '__main__':
    main()


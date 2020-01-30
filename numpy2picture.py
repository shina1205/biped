import numpy as np
import cv2

np_data = '1min'
dir = './video/{}'.format(np_data)

data = np.load(file=dir + '.npy')
print(len(data))

for i in range(len(data)):
    cv2.imwrite('{}/{}.{}'.format(dir, str(i), 'jpg'), data[i])

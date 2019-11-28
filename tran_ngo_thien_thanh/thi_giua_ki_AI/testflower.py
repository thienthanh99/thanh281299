from __future__ import print_function
# import numpy as np 
# import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mahotas
import cv2
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
np.random.seed(22)
bins = 8

features = np.loadtxt("feature-datas.txt")
labels = np.loadtxt("label-datas.txt")

#######################

V = features.T*labels
P = matrix(V.T.dot(V))

q = matrix(-np.ones((160,1)))
G = matrix(-np.eye(160))
h = matrix(np.zeros((160, 1)))
A = matrix(labels)
b = matrix(np.zeros((1,1)))
solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A.T, b)

l = np.array(sol['x'])
# print('lamda= ')
# print(l.T)

epsilon = 1e-191 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = features[:, S]
yS = labels[S]
lS = l[S]
# calculate w and b
w = XS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

# print('w = ', w.T)
# print('b = ', b)



# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

fixed_size       = tuple((500, 500))


image = cv2.imread("image_0137.jpg")

# resize the image
image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
fv_hu_moments = fd_hu_moments(image)
fv_haralick   = fd_haralick(image)
fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
ws = 0
ww = 0
for wt in range(0,160):
    ws += w[wt]*global_feature[wt]
    ww += w[wt]*w[wt]
# print((ws + b)/ ww)
mypredict = (ws + b)/ ww
print(mypredict)

label = ''
if mypredict > 0.0:
    label = 'hoa vang'
else:
    label = 'hoa trang'

# print (label)

# print(prediction)

# show predicted label on image
cv2.putText(image, str(label), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

# display the output image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
from sklearn.ensemble import RandomForestClassifier

features = np.loadtxt("feature-datas.txt")
labels = np.loadtxt("label-datas.txt")

## Train Mo hinh
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

model = RandomForestClassifier(max_depth=5, n_estimators=10)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
print("Accuracy: "+ str(100*accuracy_score(y_test,y_pred)))
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
bins             = 8
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

image = cv2.imread('image_0019.jpg')

# resize the image
fixed_size       = tuple((500, 500))
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

# scale features in the range (0-1)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(global_feature.reshape(1, -1))


# predict label of test image
prediction = model.predict(rescaled_feature.reshape(1,-1))[0]
label = 'hoa 1'

if prediction==2.0:
    label ='hoa 2'
elif prediction==3.0:
    label ='hoa 3'

print (label)
# show predicted label on image
cv2.putText(image, str(prediction), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

# display the output image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

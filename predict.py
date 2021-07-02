import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
import numpy as np

X , Y =fetch_openml('mnist_784',version = 1 , return_X_y=True)
print(pd.Series(Y).value_counts())

classes = ['0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9']
nclasses = len(classes)

Xtrain , Xtest , Ytrain ,Ytest = train_test_split(X , Y , test_size=2500 , train_size=7500 , random_state=9)

Xtrain = Xtrain/255.0
Xtest = Xtest/255.0

classifier = LogisticRegression(solver="saga" , multi_class="multinomial")
classifier.fit(Xtrain , Ytrain)

def getprediction(img):
    print("predict the digit")
    image = Image.open(img)
    imagebw = image.convert('L')
    imagebwresize = imagebw.resize((28,28),Image.ANTIALIAS)

    pixel = 20
    minpixel = np.percentile(imagebwresize ,pixel)
    imagebwresizescaled = np.clip(imagebwresize-minpixel,0,255)
    maxpixel = np.max(imagebwresize)
    imagebwresizescaled = np.asarray(imagebwresizescaled)/maxpixel
    imagearray = np.array(imagebwresizescaled).reshape(1,784)

    imagepredict = classifier.predict(imagearray)
    print(imagepredict)
    return imagepredict[0]


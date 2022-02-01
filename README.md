# Parks and Recreation Characters Face Recognition in Video

<br>

<p align="center">
  <img width="852" alt="leslie_ann_box" src="https://user-images.githubusercontent.com/46462603/150107145-92985cef-6878-4404-be8c-696e3d5c6013.png">
<img width="854" alt="leslie_box" src="https://user-images.githubusercontent.com/46462603/150107163-72e25c6f-c3d2-48cb-9437-8ead0f855e49.png">
<img width="852" alt="ron_box" src="https://user-images.githubusercontent.com/46462603/150107180-dc581cba-fda8-4399-b5ba-95b994637812.png">
<img width="854" alt="tom_box" src="https://user-images.githubusercontent.com/46462603/150107197-7300e0f8-6f19-412d-84fb-76ffe993d4f4.png">
</p>

<br>

## Worksite
The code can be found at <a href="https://colab.research.google.com/drive/1AGaILQMbOncFIG1UvSsFH7Q_OQ9ck4n8?usp=sharing" target="_blank">Parks & Recs Face Recognition on Google Colab</a>. The ipynb file is too large to be uploaded to GitHub.

## Goal
To create a classification model that is able to recognize 5 characters from a Parks & Recreation video. The 5 characters are Leslie, Ann, Ron, Tom, and Garry.

## Dataset
The training set contains 10 images from each of the 5 characters while the validation set contains 3 images from each of the 5 characters. Hence, there are altogether 50 images in the training set and 15 images in the validation set. The images are obtained from Google and cropped into squares around the characters' faces. The photos are randomly chosen. However, I tried to pick the photos that show different angles of the characters to make sure that the model is trained with as many different angles of the characters' faces as possible. 

Normally, image recognition programs will be trained with an enormous amount of data, such as what I did in the <a href="https://github.com/ZhengEnThan/Waste-Classification" target="_blank">waste image classification website</a> project. However, with <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a>, our model already has a set of pre-defined weights to use. Therefore, only a few training images are needed to fit the model to the faces of our 5 characters in Parks and Recreation.  

## Approaches
In this project, the VGG-Face model is used and loaded with the trained weights in <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a>. A convolutional neural network is created based on the layer structure defined in <a href="https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/" target="_blank">Deep Face Recognition with Keras</a>. The model has 22 layers and the last layer consists of 2622 nodes. 

The The model is later trained on the images of the 5 Parks & Recreation characters using neural network again with Haar cascade. The model has an validation accuracy of 60%.

## Comments
The model is not that effective in recognizing multiple people in the same frame. It also for some reason doesn't recognize Garry throughout the whole video.

## References
https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input
https://www.kaggle.com/acharyarupak391/vggfaceweights
https://keras.io/api/layers/reshaping_layers/zero_padding2d/
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://github.com/opencv/opencv/tree/master/data/haarcascades
https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5

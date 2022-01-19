Code: https://colab.research.google.com/drive/1AGaILQMbOncFIG1UvSsFH7Q_OQ9ck4n8?usp=sharing. The ipynb file is too large to be uploaded to GitHub

<p align="center">
  <img width="852" alt="leslie_ann_box" src="https://user-images.githubusercontent.com/46462603/150107145-92985cef-6878-4404-be8c-696e3d5c6013.png">
<img width="854" alt="leslie_box" src="https://user-images.githubusercontent.com/46462603/150107163-72e25c6f-c3d2-48cb-9437-8ead0f855e49.png">
<img width="852" alt="ron_box" src="https://user-images.githubusercontent.com/46462603/150107180-dc581cba-fda8-4399-b5ba-95b994637812.png">
<img width="854" alt="tom_box" src="https://user-images.githubusercontent.com/46462603/150107197-7300e0f8-6f19-412d-84fb-76ffe993d4f4.png">
</p>

## Goal
To create a classification model that is able to recognize 5 characters from the Parks & Recreation show. The 5 characters are Leslie, Ann, Ron, Tom, and Garry.

## Dataset
The training set contains 10 images from each of the 5 characters while the validation set contains 3 images from each of the 5 characters. Hence, there are altogether 50 images in the training set and 15 images in the validation set. The images are obtained from Google and cropped into a square around the each of the characters' face.

## Approaches
A convolutional neural network is created based on the layer structure defined in https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf. The structure is based on a research paper from the Oxford Visual Geometry Group. The model has 22 layers and 37 deep units and the last layer consists of 2622 nodes. The model is loaded with the trained VGG face weights. The model is later trained on the images of the 5 Parks & Recreation characters using neural network again with Haar cascade. The model has an accuracy of 60%.

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

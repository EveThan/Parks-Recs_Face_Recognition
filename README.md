# Parks and Recreation Characters Face Recognition in Video

<br>

These are screenshots of the resulting video frames.
<p align="center">
  <img width="852" alt="leslie_ann_box" src="https://user-images.githubusercontent.com/46462603/150107145-92985cef-6878-4404-be8c-696e3d5c6013.png">
<img width="854" alt="leslie_box" src="https://user-images.githubusercontent.com/46462603/150107163-72e25c6f-c3d2-48cb-9437-8ead0f855e49.png">
<img width="852" alt="ron_box" src="https://user-images.githubusercontent.com/46462603/150107180-dc581cba-fda8-4399-b5ba-95b994637812.png">
<img width="854" alt="tom_box" src="https://user-images.githubusercontent.com/46462603/150107197-7300e0f8-6f19-412d-84fb-76ffe993d4f4.png">
</p>

<br>

## Worksite
The code can be found at <a href="https://colab.research.google.com/drive/1AGaILQMbOncFIG1UvSsFH7Q_OQ9ck4n8?usp=sharing" target="_blank">Parks & Recs Face Recognition on Google Colab</a>. The project ipynb file is too large to be uploaded to GitHub. Similarly, the pre-trained face model weights file <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a> is too large to be uploaded here.

## Code, files, or folders needed to run the program
-  <a href="https://colab.research.google.com/drive/1AGaILQMbOncFIG1UvSsFH7Q_OQ9ck4n8?usp=sharing" target="_blank">Parks & Recs Face Recognition on Google Colab</a>
- <a href="https://github.com/ZhengEnThan/Parks-Recs_Face_Recognition/tree/main/train" target="_blank">train</a>
- <a href="https://github.com/ZhengEnThan/Parks-Recs_Face_Recognition/tree/main/val" target="_blank">val</a>
- <a href="https://github.com/ZhengEnThan/Parks-Recs_Face_Recognition/blob/main/ice_rink.mp4" target="_blank">ice_rink.mp4</a>
- <a href="https://github.com/ZhengEnThan/Parks-Recs_Face_Recognition/blob/main/haarcascade_frontalface_alt2.xml" target="_blank">haarcascade_frontalface_alt2.xml</a> which can also be found <a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml" target="_blank">here</a> 
- <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a>

## Goal
To create a classification model that is able to recognize 5 characters from a Parks & Recreation video. The 5 characters are Leslie, Ann, Ron, Tom, and Garry.

## Dataset
The training set contains 10 images from each of the 5 characters while the validation set contains 3 images from each of the 5 characters. Hence, there are altogether 50 images in the training set and 15 images in the validation set. The images are obtained from Google and cropped into squares around the characters' faces. The photos are randomly chosen. However, I tried to pick photos that show different angles of the characters to make sure that the model is trained with as many different angles of the characters' faces as possible. 

Normally, image recognition programs will be trained with an enormous amount of data, such as what I did in the <a href="https://github.com/ZhengEnThan/Waste-Classification" target="_blank">waste image classification website</a> project. However, with <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a>, our model already has a set of pre-defined weights to use. Therefore, only a few training images are needed to fit the model to the faces of our 5 characters in Parks and Recreation.  

## Approaches
In this project, the VGG-Face model is used and loaded with the trained weights in <a href="https://www.kaggle.com/acharyarupak391/vggfaceweights" target="_blank">vgg_face_weights.h5</a>. A convolutional neural network is created based on the layer structure defined in <a href="https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/" target="_blank">Deep Face Recognition with Keras</a>. The model has 22 layers and the last layer consists of 2622 nodes. 

As the original VGG-Face model was trained on the WildFace dataset, we remove the last layer of the model to fit the model onto our own dataset. Hence, our VGG-Face model will not be able to produce a prediction yet as its second last layer will only produce some numerical representations of the input images which are called embeddings. We can add one or more layers to finish the model so that it can produce predictions as probabilities. In this project, I added another 5 layers with the last layer having a Sigmoid function as its activation function to finish the model. The last layer has 5 nodes or units because it needs to produce probabilities for 5 characters for a given image input. 

In order to detect and locate multiple faces in an image, cv2.CascadeClassifier is loaded with <a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml" target="_blank">haarcascade_frontalface_alt2.xml</a>. Overall, the model has an validation accuracy of 60%. The model is later tested on the video <a href="https://github.com/ZhengEnThan/Parks-Recs_Face_Recognition/blob/main/ice_rink.mp4" target="_blank">ice_rink.mp4</a>. 

## Comments
The model is not that effective in recognizing multiple people in the same frame. It also for some reason doesn't recognize Garry throughout the whole video. Did the model somehow learn that Garry is a character that is always being ignored in the show? 😆

Will the model perform better if it is trained on more photos of these characters?

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

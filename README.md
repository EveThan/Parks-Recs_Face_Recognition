

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
https://medium.com/analytics-vidhya/how-to-implement-face-recognition-using-vgg-face-in-python-3-8-and-tensorflow-2-0-a0593f8c95c3
https://keras.io/api/layers/reshaping_layers/zero_padding2d/
https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/backend/eval.html
https://ai.stackexchange.com/questions/31675/what-is-better-to-use-early-stopping-model-checkpoint-or-both
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://github.com/opencv/opencv/tree/master/data/haarcascades
https://stackoverflow.com/questions/50963283/python-opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
https://dsp.stackexchange.com/questions/28283/should-i-use-cv-haar-scale-image-while-using-lbp-cascadeclassifier

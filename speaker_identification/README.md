# Dominant Speaker identification in vedios

### Problem: Given a vedio ,need to find which one of these six speaker is present there ?

1. Atul Khatri
2. Flute Raman
3. Sadhguru
4. Sandeep Mahshwari
5. Sorabh Pant
6. Shailendra
7. None of these 6

## Dataset Preparation:

-Downloaded 20-25 videos (720p) of each speaker from youtube (used different videos for frame extraction rather than extracting more frames from same video) 

-Extract frames using VideoCapture() and read() opencv functions with suitable FPS and put into respective folders. 

-First we tried Haar CascadeClassifier in opencv for face detection in frames. (Advantage: It's fast., Disadvantae: Its facing problem for face of person with beared like sadhguru), In our case need more info than only the face.

-So we used YOLO to get the frames which contains class person, In case of multiple person detected in a frame we extract the person having probability > .98 assuming that the dominant person (speaker) will have a good coverage in the frame. In case of frames with only one person we need to decide that it is person of our interest (speaker with one of the 6 classes or any other person (from audience)), For deciding this we take 2-Norm of image pixels and takes difference with mean of the 2-Norm of all images, if its less than 10000, we retain this frame because it contains person of our interest. 

## Training and Testing:

-We tried tranfer learning with VGG16, Resnet50 and Inceptionv3 networks with only classifiers trained and with last two layers (fc1 and fc2) also trained.


| Model | description | Trainig data | Training examples | epoch | Val. acc | Test acc_1 | Test acc_1 (cropped testdata) | Test acc_2 | Test acc_2 (cropped testdata) |

| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| a | a | a | a | a | a | a | a | a | a |


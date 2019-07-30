# Dominant Speaker identification in videos

### Problem: Given a video, need to find which one of these six speaker is present there ?

Classes:

1. Atul Khatri
2. Flute Raman
3. Sadhguru
4. Sandeep Mahshwari
5. Sorabh Pant
6. Shailendra
7. None of these 6

## Dataset Preparation:

-Downloaded 20-25 videos (720p) of each speaker from youtube (used different videos for frame extraction rather than     extracting more frames from same video) 

-Extract frames using VideoCapture() and read() opencv functions with suitable FPS and put into respective folders. 

-First we tried Haar CascadeClassifier in opencv for face detection in frames. (Advantage: It's fast., Disadvantae: It's  facing problem for face of person with beared like sadhguru), In our case we need more info than only the face to recognize a person.

-So we used YOLO to get the frames which contains class person, In case of multiple person detected in a frame we extract the person having probability > .98 assuming that the dominant person (speaker) will have a good coverage in the frame. In case of frames with only one person we need to decide that it is person of our interest (speaker with one of the 6 classes or any other person (from audience)), For deciding this we take 2-Norm of image pixels and takes difference with mean of the 2-Norm of all images, if its less than 10000, we retain this frame because it contains person of our interest. 

## Training and Testing:

-We tried tranfer learning with VGG16, Resnet50 and Inceptionv3 networks with only classifiers trained and with last two layers (fc1 and fc2) also trained.

| Model | description | Trainig data | Training examples | epoch | Val. acc | Test acc_1 | Test acc_1 (cropped testdata) |Test acc_2 | Test acc_2 (cropped testdata) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| vgg16_0_30_0.h5 (file: vgg16_0_30_0.py ) | Only classifier trained | Frames | 8297 | 30 | 99.52% | 62.05% | 60.6% | 62.05% | 61.21% |
| Vgg16 (vgg16_1_30_0.h5) |  Last two layers trained  |  Frames  |  8297  | 30 | 99.54%  | 57.47%  | 44.8% | 59.44%  | 45.36% | 
| Resnet50(resnet50_0_30_0.h5) |  Only classifier trained |  Frames  |  8297  | 30 | 99.14% | 59.69% | 67.36% | 61.16% | 67.08% |
| Resnet50 (resnet50_1_30_0.h5) |  Last two layers trained |  Frames  |  8297  | 20 | 99.25% | 57.94% | 63.79% | 59.94% |     64.83% |
| Inceptionnetv3(Inceptionnetv3_1_30_0.h5) | Last two layers trained | Frames | 8297 | 30 | 99.80% | 56.88% | 60.03% | 62.36% | 60.50% |
| Inceptionnetv3(Inceptionnetv3_1_50_0.h5) | Last two layers trained | Frames | 8297 | 50 | 99.45% | 61.40% | 65.03% | 62.58% | 65.42% |
| Vgg16(vgg16_0_30_1.h5) | Only classifier trained | Cropped frames | 7623 | 30 | 99.13% | 69.49% | 65.00% | 71.74% | 64.24% |
| Vgg16(vgg16_1_30_1.h5) | Last two layers trained | Cropped frames | 7623 | 30 | 99.87% | 58.79% | 62.16% | 60.96% | 63.31% |
| Resnet50(resnet50_0_30_1.h5) | Only classifier trained | Cropped frames | 7623 | 30 | 99.73% | 71.60% | 75.17% | 73.99% |   75.98% |
| Resnet50(resnet50_1_30_1.h5) | Last two layers trained | Cropped frames | 7623 | 30 | 99.54% | 67.92% | 71.45% | 70.87% | 72.81% |
| Resnet50(resnet50_0_50_1.h5) | Only classifier trained | Cropped frames | 7623 | 30 | 99.35% | 67.61% | 72.33% | 70.65% | 74.21% |
| inceptionnetv3(inceptionnetv3_1_30_1.h5) | Last two layers trained | Cropped frames | 7623 | 30 | 99.61% | 61.07% | 69.13% | 61.40% | 69.55% |






 






# Ekam

This is a Face recognition and emotion classification program written in python.
The program uses the KDEF http://www.emotionlab.se/resources/kdef dataset to train the emotion classification.

## Software Dependencies

- Python 2.7
- OpenCV 2.4

## Outline of the project

This project consists of three parts, 
1. Creating a dataset of the faces thats to be recognized
2. Training the model
3. Face recognition

## How to run the file

1. Store the KDEF dataset in a folder called KDEF
2. Create a folder called Face_name and store the name of the persons in this folder in the format AF01AMS.JPG, where AM is the key for a value of the name of the person
3. Run Emotions.train.py first to create a model for emotions classification
4. Run Faces_train.py to train the face classification algorithm
5. then execute people_emotions.py for real time emotion classification and recognizing peoples faces 


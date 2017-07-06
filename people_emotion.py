import cv2
import sys
import time
import requests


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS,10)


cv2.namedWindow('Final',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Final', 1000,1000)
'''Face REcognition Variables
    ppl is a list of names, fishface is an object fot face recogniser,
    MODEL_FILE is the location where the model is stored
    '''
ppl = ['Rakshith','Appa','Amma','Patti','Anirudh','Gutti','Noopy']
fishface_ppl = cv2.face.createLBPHFaceRecognizer() 
print ('loading peoples model')
MODEL_FILE_ppl="face_identifier.model"
fishface_ppl.load(MODEL_FILE_ppl)
prediction_print_ppl = []


emotions = ['afraid','angry','disgusted','happy','neutral','sad','surprised']
fishface_emo = cv2.face.createLBPHFaceRecognizer() 
print ('loading emotions model')
MODEL_FILE_EMO="emotions_identifier.model"
fishface_emo.load(MODEL_FILE_EMO)
prediction_print_emo = []
previous_time = 0



def main():
    while (True):
        #cv2.imshow('raw',frame)
        # Capture frame-by-frame

        ert, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        global previous_time

        if time.time() - previous_time > 2: 
            
            recognise_ppl(gray,faces)
            recognise_emotions(gray,faces)
            # json_peop_emo = name_json(prediction_print_ppl,prediction_print_emo)
            # req = requests.post('http://127.0.0.1:5000/vid',json=json_peop_emo)
            # print ("status:")
            # print(req.status_code)
            previous_time = time.time()  

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w+10, y+h+10), (0, 255, 0), 2)
        for  pred,x,y,w,h,_ in prediction_print_ppl:
            # Display the resulting frame
            cv2.putText(frame,pred,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        # for  pred,x,y,w,h,_ in prediction_print_emo:
        #     # Display the resulting frame
        #     cv2.putText(frame,pred,(x,y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)


        Final_resized_image = cv2.resize(frame, (940, 800))    
        cv2.imshow('Final', Final_resized_image)

       # time.sleep(60)

        #cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
def name_json(people,emotion):

    data_people = []
    for i,(ppl_id,_,_,_,_,_) in enumerate(people):
        data_people.append({'name':ppl_id,'emotion':emotion[i][0], 'emoconf':emotion[i][5]})
    data = {'people':data_people}
    print data
    return data

def recognise_ppl(image,faces):

    global prediction_print_ppl
    prediction_print_ppl = []
    for (x, y, w, h) in faces:
        img = image[y:y+h+10,x:x+w+10]
        resized_image = cv2.resize(img, (128, 128))
        pred, conf = fishface_ppl.predict(resized_image)
        #print "Prediction ",ppl[pred]," Confidance ",conf
        prediction_print_ppl.append((ppl[pred],x,y,w,h,float(conf)))
      
 


def recognise_emotions(image,faces):
    global prediction_print_emo


    prediction_print_emo = []

    for (x, y, w, h) in faces:
        img = image[y:y+h+10,x:x+w+10]
        resized_image = cv2.resize(img, (128, 128))
        pred, conf = fishface_emo.predict(resized_image)            
        #print "Prediction ",emotions[pred]," Confidance ",conf
        prediction_print_emo.append((emotions[pred],x,y,w,h,float(conf)))        
         


if __name__ == "__main__":
    main()
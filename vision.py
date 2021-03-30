import streamlit as st
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

st.header("Artificial Vision for indoor navigation")
st.subheader("Detecting doors,furnitures, stairs beforehand and prevent accidents")
loaded_model=load_model("vision1.h5")
FRAME_WINDOW = st.image([])
webcam = cv2.VideoCapture(0)
if st.checkbox("Run"):
    
    webcam = cv2.VideoCapture(0)

    classes = ['stairs','furniture',"door"]
    

    # loop through frames
    while webcam.isOpened():


            status, frame = webcam.read()
            face_crop=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            

            face_crop = cv2.resize(face_crop, (300,300))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)


            conf = loaded_model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]


            idx = np.argmax(conf)
            label = classes[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)




            face_cropped=cv2.putText(frame, label, (140, 60),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            

            iml=face_cropped
            FRAME_WINDOW.image(iml)


           
webcam.release()
cv2.destroyAllWindows()  

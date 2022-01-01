import streamlit as st
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
import io
import os


# Function for prediction in images
def image_detect_and_predict(frame, masknet):
    faces, locs, preds = [], [], []

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img)
    for result in results:
        if result['confidence'] > 0.5:
            box = result['box']

            face = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((box[0], box[1], box[0] + box[2], box[1] + box[3]))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = masknet.predict(faces, batch_size=32)
    return (locs, preds)


# Function for detection and displaying resuting images
def image_predict(frame, masknet):

    (locs, preds) = image_detect_and_predict(frame, maskNet)


    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (incorrect, mask, withoutMask) = pred
    
        if max(pred) == mask:
            label = "With Mask"
            color = (0, 255, 0)
        elif max(pred) == withoutMask:
            label = "Without Mask"
            color = (0, 0, 255)
        else:
            label = "Wearing Mask Incorrectly"
            color = (0, 255, 255)
    
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(pred) * 100)
    
        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
        # show the output frame
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Predictions', width=720)



def video_detect_and_predict(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104.0, 177.0, 123.0))

    # pass the blob into model
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence of face detection
        confidence = detections[0, 0, i, 2]

        # retain only detection for which confidence>0.5
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (incorrect, mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        if max(pred) == mask :
            label = "With Mask"
            color = (0, 255, 0)
        elif max(pred) == withoutMask:
            label = "Without Mask"
            color = (0, 0, 255)
        else:
            label = "Wearing Mask Incorrectly"
            color = (0, 255, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(pred) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return frame






st.set_page_config(page_title="Face Mask Detection System", page_icon="https://lensec.com/wp-content/uploads/2021/03/Mask-Detection-Icon.png")
st.title('Face Mask Detection System')

cwd = os.path.dirname(__file__)
prototxtPath = os.path.join(cwd,"face_detection_dnn\\deploy.prototxt.txt")
weightsPath = os.path.join(cwd,"face_detection_dnn\\res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(cwd, 'face_mask_detector.model'))


rad = st.sidebar.radio("Navigation", ["Home", "About Model"])

if rad == "Home":
    st.markdown(""" ## Chose detction options from sidebar: """)
    detection_choice = st.sidebar.selectbox("Detection Preference", ["Detect Mask in Image", "Detect Mask in Video"])
    if detection_choice=="Detect Mask in Image":
        img = st.file_uploader("Upload a image", type=['png', 'jpg', 'jpeg'])
        if img != None:

            st.image(img, caption="Original Image", width=720)
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            image_predict(frame, maskNet)

    if detection_choice == "Detect Mask in Video":
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        camera_choice = st.selectbox("Video Options",["Upload Video","Open WebCam"],index = 1)
        if camera_choice == "Open WebCam":
            run = st.checkbox('Start WebCam')
            FRAME_WINDOW = st.image([])

            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            while run:
                ret, frame = cam.read()
                frame = video_detect_and_predict(frame, faceNet, maskNet)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
            else: cam.release()

        elif camera_choice=="Upload Video":
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
            temporary_location = False

            if uploaded_file is not None:
                g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
                temporary_location = "testout_simple.mp4"

                with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                    out.write(g.read())  ## Read bytes into file

                # close file
                out.close()


            @st.cache(allow_output_mutation=True)
            def get_cap(location):
                video_stream = cv2.VideoCapture(str(location))

                # Check if camera opened successfully
                if (video_stream.isOpened() == False):
                    print("Error opening video  file")
                return video_stream


            image_placeholder = st.empty()

            if temporary_location:
                while True:
                    # here it is a CV2 object
                    video_stream = get_cap(temporary_location)
                    # video_stream = video_stream.read()
                    ret, frame = video_stream.read()
                    frame = video_detect_and_predict(frame, faceNet, maskNet)
                    image_placeholder.image(frame, channels="BGR", use_column_width=True)

                    cv2.destroyAllWindows()
                video_stream.release()


                cv2.destroyAllWindows()

        

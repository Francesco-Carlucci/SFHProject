import cv2
import mediapipe as mp

import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if __name__ == '__main__':
    prova1="./prova1.jpg"


    cap = cv2.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
    empty, frame = cap.read()  # return a single frame in variable `frame`
    while (True):
        empty, frame = cap.read()
        if not empty:
            continue
        cv2.imshow('captured img', frame)
        if cv2.waitKey(10) & 0xFF == ord(' '):
            #cv2.imwrite('prova1', frame)
            cv2.destroyAllWindows()
            break

    cap.release()
    # images_namelist = os.listdir(path)
    with mp_hands.Hands(static_image_mode=True,max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        img=cv2.flip(frame,1)  #flip horizontally
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #pass to RGB (cv2 internal representation is BGR)

        output=hands.process(img)

        print('Handedness:', output.multi_handedness)
        print('shape: ',img.shape)
        h,w,_=img.shape
        imgPoints=img.copy() #duplica immagine
        #print('result: ', output.multi_hand_landmarks)
        i=0
        print("output: ",output.palm_detections[0].location_data.relative_bounding_box)
        xmin=output.palm_detections[0].location_data.relative_bounding_box.xmin*w
        ymin=output.palm_detections[0].location_data.relative_bounding_box.ymin*h
        width=output.palm_detections[0].location_data.relative_bounding_box.width*w
        height=output.palm_detections[0].location_data.relative_bounding_box.height*h
        BBpoints=np.array([[xmin,ymin],[xmin+width,ymin],[xmin,ymin+height],[xmin+width,ymin+height]])
        BBLargePoints=np.array([[xmin-width,ymin-height],[xmin+width,ymin-height],[xmin-width,ymin+height],[xmin+width,ymin+height]])
        imgBB=img.copy()
        plt.imshow(imgBB)
        plt.scatter(BBpoints[:,0],BBpoints[:,1])
        plt.scatter(BBLargePoints[:, 0], BBLargePoints[:, 1],c='red')
        plt.show()

        for punti in output.multi_hand_landmarks:
            i+=1
            #print('punti ', i, ': ', punti)
            #punti.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w #coordinata x del dito indice normalizzata
            mp_drawing.draw_landmarks(
                imgPoints,
                punti,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            #cv2.imwrite('prova1Punti.png',cv2.flip(imgPoints,1))
            cv2.imshow('img1', cv2.cvtColor(imgPoints,cv2.COLOR_RGB2BGR))  # display the captured image
            cv2.waitKey(10000)
            #cv2.imwrite('./'+'provaPunti3'+'.png',imgPoints)
        '''
        if output.multi_hand_world_landmarks:
            for punti3D in output.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                punti3D, mp_hands.HAND_CONNECTIONS, azimuth=5)
        '''

    print('Prova video')

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=5,
                        model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            notEmpty, img=cap.read()
            if not notEmpty:
                continue

            img.flags.writeable=False
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            output=hands.process(img)

            img.flags.writeable=True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if output.multi_hand_landmarks:
                for points in output.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,points,mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                if output.palm_detections:
                    xmin = output.palm_detections[0].location_data.relative_bounding_box.xmin * w
                    ymin = output.palm_detections[0].location_data.relative_bounding_box.ymin * h
                    width = output.palm_detections[0].location_data.relative_bounding_box.width * w
                    height = output.palm_detections[0].location_data.relative_bounding_box.height * h
                    BBpoints = np.array(
                        [[xmin, ymin], [xmin + width, ymin], [xmin, ymin + height], [xmin + width, ymin + height]])
                    for p in BBpoints:
                        img=cv2.circle(img,(int(p[0]),int(p[1])),radius=5, color=(0, 0, 255))

            cv2.imshow('MediaPipe Hands', cv2.flip(img, 1))
            if cv2.waitKey(5) & 0xFF == 27:  #27 is ascii for ESC, 0xFF seleziona il byte meno significativo
                break
        cap.release()


    cap = cv2.VideoCapture(0)
    cap.set( cv2.VIDEOWRITER_PROP_DEPTH,1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output.avi', fourcc, 20.0, (640, 480))
    #out=cv2.VideoWriter.open('./output.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error!")
            break
        (B, G, R) = cv2.split(frame)

        #frame = cv2.flip(frame, 0)
        # write the flipped frame
        cv2.imshow('frame',frame)
        cv2.waitKey(20)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

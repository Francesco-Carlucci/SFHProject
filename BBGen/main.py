
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from threading import Lock

import time
import mediapipe
import cv2
import numpy as np
import os


#modificare in calcola centro->calcola BB massime dim.

def findBBfromPoints(points):
    xmin, ymin = np.min(points, 0).astype(int)
    xmax, ymax = np.max(points, 0).astype(int)
    return xmin,ymin,xmax,ymax

def getMaxBB(xmin,ymin,xmax,ymax,maxw,maxh,w,h):
    center = [np.round((xmax - xmin) / 2)+ xmin, np.round((ymax - ymin) / 2) + ymin]
    newxmin = int(np.round(center[0] - maxw / 2))
    newymin = int(np.round(center[1] - maxh / 2))
    newxmax = int(np.round(center[0] + maxw / 2))
    newymax = int(np.round(center[1] + maxh / 2))

    bordersize = max(0 - newxmin, 0 - newymin, newxmax - w, newymax - h, 0)
    newxmin += bordersize
    newxmax += bordersize
    newymin += bordersize
    newymax += bordersize

    return newxmin,newymin,newxmax,newymax,bordersize

def resizeSameRatio(img,height,width):
    h, w, _ = img.shape
    if h>w:
        bordersize = h-w

        img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=0,
            left=0,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    elif w>h:
        bordersize= w-h
        img = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    img=cv2.resize(img,(width,height))
    return img

def cropBB(detectedHand,w,h,maxw,maxh,frame):  #da centro,w,h,rotation (rad) ottiene la bb non ruotata e taglia
    xc = detectedHand.x_center * w
    yc = detectedHand.y_center * h
    C = np.array([xc, yc])
    width = detectedHand.width * w
    height = detectedHand.height * h
    teta = detectedHand.rotation
    hvet = np.array([-np.sin(teta), np.cos(teta)])
    wvet = np.array([np.cos(teta), np.sin(teta)])
    """
    a = np.round(C - hvet * maxh / 2 - wvet * maxw / 2)
    b = np.round(C - hvet * maxh / 2 + wvet * maxw / 2)
    d = np.round(C + hvet * maxh / 2 + wvet * maxw / 2)
    e = C + hvet * maxh / 2 - wvet * maxw / 2
    """
    a = np.round(C - hvet * height / 2 - wvet * width / 2)
    b = np.round(C - hvet * height / 2 + wvet * width / 2)
    d = np.round(C + hvet * height / 2 + wvet * width / 2)
    e = C + hvet * height / 2 - wvet * width / 2

    points = np.array([a, b, d, e])  # trova rettangolo non ruotato che contiene la bounding box
    xmin = min(points[:, 0])
    xmax = max(points[:, 0])
    ymin = min(points[:, 1])
    ymax = max(points[:, 1])

    bordersize = int(max(0 - xmin, 0 - ymin, xmax - w, ymax - h,0))
    print('border size in px: ', bordersize)
    xmin += bordersize
    xmax += bordersize
    ymin += bordersize
    ymax += bordersize

    frame = cv2.copyMakeBorder(
        frame,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    """ #controlla punti
    cv2.circle(frame, (int(a[0]), int(a[1])), radius=10, color=(0, 0, 255))
    cv2.circle(frame, (int(b[0]), int(b[1])), radius=10, color=(0, 0, 255))
    cv2.circle(frame, (int(d[0]), int(d[1])), radius=10, color=(0, 0, 255))
    img2 = cv2.circle(frame, (int(e[0]), int(e[1])), radius=10, color=(0, 0, 255))
    cv2.imshow('punti', img2)  #cv2.resize(img2, (newW, newH))
    cv2.waitKey()
    """
    cv2.circle(frame, (int(xmin), int(ymin)), radius=5, color=(0, 0, 255))
    cv2.circle(frame, (int(xmax), int(ymax)), radius=5, color=(0, 0, 255))
    cv2.imshow('rettangolo',frame)
    cv2.waitKey()

    #tetaDeg=math.degrees(teta)   #ruota l'immagine, perde informazione gesti
    #M = cv2.getRotationMatrix2D(C, tetaDeg+180, 1)
    #img_rot = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    #cropped = frame[int(d[1]):int(a[1]), int(b[0]):int(e[0]), :]
    cropped=frame[int(ymin):int(ymax),int(xmin):int(xmax)]

    #cv2.imshow('rotated',cv2.resize(img_rot,(img_rot.shape[1]//3, img_rot.shape[0]//3)))
    #cv2.imshow('cropped', cv2.resize(cropped, (cropped.shape[1], cropped.shape[0])))
    #cv2.waitKey()
    return cropped

datasetPath='../GesRec/datasets/jester_kaggle/archive/Train' #../GesRec/datasets/jester_kaggle/archive/Train  #../20bn-jester-v1

lock=Lock()
videoCnt=0

@profile
def mainTask(videoList):
    global videoCnt

    resultDatasetPath='./jester'

    mp_hands = mediapipe.solutions.hands  # seleziona hand dalla libreria mediapipe SPOSTARE in schedule

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.4) as handDet:
        for videoPath in videoList:
            video=[]
            coordinates=[]   #boundng box coordinates for each frame of the video

            with lock:
                videoCnt += 1
                print('processing: ',videoPath)
                if videoCnt%50==0:
                    print('{:.4f}'.format(videoCnt/len(videoList)*100),'%')
            resultPath = os.path.join(resultDatasetPath,videoPath)
            if not os.path.exists(resultPath):
                os.mkdir(resultPath)
            else:
                print(videoPath,' already processed')
                continue
            videoPath=os.path.join(datasetPath,videoPath)
            imgList=os.listdir(videoPath)

            maxw=0
            maxh=0
            frameCnt=0
            for imgPath in imgList:       #per ogni video calcola le massime base e altezza

                frame=cv2.imread(os.path.join(videoPath,imgPath))

                if frameCnt==0:
                    h, w, _ = frame.shape

                frameCnt += 1

                frame.flags.writeable=False
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                output=handDet.process(frame)

                if output.multi_hand_landmarks!=None:
                    #for detectedHand in output.hand_rects:
                    for points in output.multi_hand_landmarks:
                        points=[[p.x*w,p.y*h] for p in points.landmark]   #estrae tutti i ounti della mano
                        xmin,ymin,xmax,ymax=findBBfromPoints(points)       #calcola rettangolo che li contiene tutti

                        video.append(frame)
                        coordinates.append([xmin,ymin,xmax,ymax])

                        if xmax-xmin>maxw:                          #calcola massime base e altezza per tutto il video
                            maxw=xmax-xmin
                        if ymax-ymin>maxh:
                            maxh=ymax-ymin

            maxw=maxw*1.4           #applica margine del 40%
            maxh*=1.4

            imgIndex = 0
            for i,frame in enumerate(video):
                xmin, ymin, xmax, ymax=coordinates[i]
                newxmin, newymin, newxmax, newymax, bordersize = getMaxBB(xmin, ymin, xmax, ymax, maxw, maxh, w, h)
                frame = cv2.copyMakeBorder(
                    frame,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                cropped = frame[newymin:newymax, newxmin:newxmax]

                cropped = resizeSameRatio(cropped, 112, 112)
                imgIndex += 1
                imgName = '{:05d}.jpg'.format(imgIndex)
                cv2.imwrite(os.path.join(resultPath, imgName), cropped)

    cv2.destroyAllWindows()

def schedule():
    nWorkers=os.cpu_count()
    start=time.time()
    executor = ThreadPoolExecutor(max_workers=nWorkers)
    videoList = os.listdir(datasetPath)[0:10]
    videosPerThread=len(videoList)//nWorkers

    futures=[executor.submit(mainTask,videoList[i*videosPerThread:(i+1)*videosPerThread]) for i in range(nWorkers)]
    futures.append(executor.submit(mainTask,videoList[nWorkers*videosPerThread:len(videoList)]))
    #for future in as_completed(futures):
        #data=future.result()
        #print(data)
    wait(futures)
    print('durata: ',time.time()-start)

if __name__ == '__main__':
    schedule()
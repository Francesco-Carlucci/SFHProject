import os
import torch
import numpy as np
import json
import cv2
import mediapipe
import pdb
from torch.nn import functional
import time
#import ffmpeg

from PIL import Image  #controllare

import sys
#sys.path.insert(1, '/Users/utente/Desktop/SFHProject/minTrain')
from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from utils import Queue

def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))

opt = parse_opts_online()


def load_clf(opt):

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    #classifier = torch.nn.DataParallel(classifier)
    #classifier = torch.nn.DataParallel(classifier)
    #classifier = classifier.cuda()
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path,map_location=torch.device("cpu"))
        classifier.load_state_dict(checkpoint['state_dict'])

    #print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of classifier trainable parameters: ", pytorch_total_params)

    return classifier

if __name__ == '__main__':
    resultPath='./testResult/'
    classifier=load_clf(opt)

    norm_method = Normalize(opt.mean, [1, 1, 1])

    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
    ])

    fps=""
    if opt.video=='webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap=cv2.VideoCapture(opt.video) #opt.video
        """
        meta_dict = ffmpeg.probe(opt.video)
        
        rotateCode = None
        if(int(meta_dict['streams'][0]['tags']['rotate']):
            if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                rotateCode = cv2.ROTATE_180
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
        """

    num_frame=0 #per la clip del clf
    clip=[]

    active_index = 0  # conta il numero di frame attivi, in cui il classificatore ha girato
    active=False
    prev_active = False
    finished_prediction=None
    pre_predict = False

    classifier.eval()
    cum_sum=np.zeros(opt.n_classes_clf,)
    clf_selected_queue = np.zeros(opt.n_classes_clf, )  #
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
    passive_count=0

    results=[]

    mp_hands=mediapipe.solutions.hands  #seleziona hand dalla libreria mediapipe
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as handDet:

        out = cv2.VideoWriter(resultPath+'result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (270,480))
        i=0
        while cap.isOpened():
            start=time.time()
            ret,frame=cap.read()

            #frame=cv2.rotate(frame,rotateCode)

            i+=1
            if not ret:
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                #frameShow = cv2.resize(frame, (480, 320))
                predictedStr = " Result: " + str(best1)
                print('predicted classes: \t', best1)
                cv2.putText(frameShow, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1,cv2.LINE_AA)
                cv2.putText(frameShow, predictedStr, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1,cv2.LINE_AA)
                out.write(frameShow)
                break

            h,w,_=frame.shape
            frame.flags.writeable = False #forza il passagio by reference, senza copiare
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            output = handDet.process(frame)
            #print(output.palm_detections)

            frame.flags.writeable = True  #rimette a posto l'immagine per visualizzarla
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with torch.no_grad():
                if output.multi_hand_landmarks:  #non palm_detections

                    if output.multi_hand_landmarks:
                        for points in output.multi_hand_landmarks:
                            mp_drawing = mediapipe.solutions.drawing_utils
                            mp_drawing_styles = mediapipe.solutions.drawing_styles
                            mp_drawing.draw_landmarks(
                                frame, points, mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                    """
                    print("hand detected",active_index)
                     #print detected bounding box
                    xmin = output.palm_detections[0].location_data.relative_bounding_box.xmin * w
                    ymin = output.palm_detections[0].location_data.relative_bounding_box.ymin * h
                    width = output.palm_detections[0].location_data.relative_bounding_box.width * w
                    height = output.palm_detections[0].location_data.relative_bounding_box.height * h
                    
                    BBpoints = np.array(
                        [[xmin, ymin], [xmin + width, ymin], [xmin, ymin + height], [xmin + width, ymin + height]])
                    for p in BBpoints:
                        img = cv2.circle(frame, (int(p[0]), int(p[1])), radius=5, color=(0, 0, 255))
                    """
                    #crea la sliding window per il classificatore
                    if num_frame==0:
                        #cur_frame = cv2.resize(frame, (320, 240))
                        cur_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cur_frame = cur_frame.convert('RGB')
                        for i in range(opt.sample_duration):
                            clip.append(cur_frame)
                        clip = [spatial_transform(img) for img in clip]
                    clip.pop(0)
                    #_frame = cv2.resize(frame, (320, 240))
                    _frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    _frame = _frame.convert('RGB')
                    _frame = spatial_transform(_frame)
                    clip.append(_frame)
                    im_dim = clip[0].size()[-2:]
                    try:
                        test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
                    except Exception as e:
                        pdb.set_trace()
                        raise e
                    inputs = torch.cat([test_data], 0).view(1, 3, opt.sample_duration, 112, 112)
                    num_frame += 1
                    """
                    if opt.modality_clf == 'Depth':
                        inputs_clf = inputs[:, -1, :, :, :].unsqueeze(1)  # Depth
                    elif opt.modality_clf == 'RGB':
                        inputs_clf = inputs[:, :, :, :, :]
                    """
                    inputs_clf = torch.Tensor(inputs.numpy()[:, :, ::1, :, :]) #only RGB
                    #classificazione
                    outputs_clf = classifier(inputs_clf)
                    outputs_clf = functional.softmax(outputs_clf, dim=1) #softmax normalization by pytorch
                    outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
                    #print('output:',outputs_clf)
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count = 0 #flag that count number of frame without hand

                    if opt.clf_strategy == 'raw':
                        clf_selected_queue = outputs_clf
                    elif opt.clf_strategy == 'median':
                        clf_selected_queue = myqueue_clf.median
                    elif opt.clf_strategy == 'ma':
                        clf_selected_queue = myqueue_clf.ma
                    elif opt.clf_strategy == 'ewma':
                        clf_selected_queue = myqueue_clf.ewma

                else:
                    outputs_clf = np.zeros(opt.n_classes_clf, )
                    # Push the probabilities to queue
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count += 1


            if passive_count >= opt.det_counter:
                active = False
                #num_frame=0
            else:
                active = True  #disattiva classificazione dopo det_counter frame senza mano
                            # se det_counter=1 (default) ferma la classificazione al primo frame
                            #in cui la mano non viene rilevata
            if active:
                active_index+=1
                cum_sum = ((cum_sum * (active_index - 1)) + (
                        weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
                #cum_sum = ((cum_sum * (active_index - 1)) + (1.0 * clf_selected_queue)) / active_index
                best2, best1 = tuple(cum_sum.argsort()[-2:])  #seleziona le due classi più probabili
                if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre: #differenza tra le due classi più probabili
                    finished_prediction = True                      #superiore alla soglia=>il modello ha prepredetto
                    pre_predict = True

            else:
                active_index=0

            if active == False and prev_active == True: #se
                finished_prediction = True
            elif active == True and prev_active == False:
                finished_prediction = False

            #best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
            #print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                #(active_index * opt.stride_len) + opt.sample_duration_clf))
            if finished_prediction == True:

                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if cum_sum[best1] > opt.clf_threshold_final:
                    results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                    print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                     (active_index * opt.stride_len) + opt.sample_duration_clf))
                finished_prediction = False
                prev_best1 = best1
                cum_sum=np.zeros(opt.n_classes_clf,)
            prev_active = active
            elapsedTime = time.time() - start
            fps = "(Playback) {:.1f} FPS".format(1 / elapsedTime)

            if len(results) != 0:
                predicted = np.array(results)[:, 1]
                prev_best1 = -1
            else:
                predicted = []

            print('predicted classes: \t', predicted)
            predictedStr = " Result: "
            for i in range(len(predicted)//11):
                predictedStr += str(predicted[i*11:(i+1)*11])+'\n'
            predictedStr += str(predicted[i * 11:])

            frameShow = cv2.resize(frame, (270,480))
            cv2.putText(frameShow, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
            if(len(predicted)>0):
                cv2.putText(frameShow, "Result: "+str(predicted[-1]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

            out.write(frameShow)

            cv2.imshow("Result", frameShow)
            #cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
            if cv2.waitKey(2) & 0xFF==27:
                break
        cap.release()
    out.release()

import os
import json

class0List=[
'class0/01subject3_r0',
'class0/01subject3_r1',
'class0/01subject4_r2',
'class0/01subject7_r0',
'class0/03subject3_r0',
'class0/03subject3_r1',
'class0/04subject3_r0',
'class0/04subject3_r1',
'class0/05subject3_r0',
'class0/05subject3_r1',
'class0/06subject3_r0',
'class0/06subject3_r1',
'class0/07subject3_r0',
'class0/07subject3_r1',
'class0/08subject3_r0',
'class0/08subject3_r1',
'class0/09subject3_r0',
'class0/09subject3_r1',
'class0/10subject3_r0',
'class0/10subject3_r1',
'class0/11subject3_r0',
'class0/11subject3_r1',
'class0/12subject3_r0',
'class0/12subject3_r1',
'class0/13subject3_r0',
'class0/13subject3_r1',
'class0/14subject3_r0',
'class0/14subject4_r0',
'class0/15subject3_r0',
'class0/15subject3_r1',
'class0/16subject3_r0',
'class0/16subject3_r1',
'class0/16subject11_r2',
'class0/17subject3_r0',
'class0/18subject3_r0',
'class0/18subject3_r1',
'class0/19subject3_r0',
'class0/19subject3_r1',
'class0/20subject3_r0',
'class0/20subject3_r1',
'class0/21subject3_r0',
'class0/21subject3_r1',
'class0/22subject3_r0',
'class0/22subject3_r1',
'class0/23subject3_r0',
'class0/24subject3_r0',
'class0/24subject3_r1',
'class0/25subject4_r1']

class0Annotations=[
{'label': '0', 'start_frame': '146', 'end_frame': '226'},
{'label': '0', 'start_frame': '160', 'end_frame': '240'},
{'label': '0', 'start_frame': '144', 'end_frame': '224'},
{'label': '0', 'start_frame': '110', 'end_frame': '190'},
{'label': '0', 'start_frame': '156', 'end_frame': '236'},
 {'label': '0', 'start_frame': '138', 'end_frame': '218'},
 {'label': '0', 'start_frame': '135', 'end_frame': '215'},
{'label': '0', 'start_frame': '140', 'end_frame': '220'},
 {'label': '0', 'start_frame': '132', 'end_frame': '212'},
 {'label': '0', 'start_frame': '132', 'end_frame': '212'},
 {'label': '0', 'start_frame': '140', 'end_frame': '220'},
{'label': '0', 'start_frame': '130', 'end_frame': '210'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
 {'label': '0', 'start_frame': '135', 'end_frame': '215'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
 {'label': '0', 'start_frame': '137', 'end_frame': '217'},
 {'label': '0', 'start_frame': '135', 'end_frame': '215'},
{'label': '0', 'start_frame': '137', 'end_frame': '217'},
{'label': '0', 'start_frame': '142', 'end_frame': '222'},
{'label': '0', 'start_frame': '135', 'end_frame': '215'},
{'label': '0', 'start_frame': '134', 'end_frame': '214'},
{'label': '0', 'start_frame': '140', 'end_frame': '220'},
{'label': '0', 'start_frame': '142', 'end_frame': '222'},
 {'label': '0', 'start_frame': '145', 'end_frame': '225'},
{'label': '0', 'start_frame': '147', 'end_frame': '227'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
{'label': '0', 'start_frame': '125', 'end_frame': '205'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
{'label': '0', 'start_frame': '132', 'end_frame': '212'},
{'label': '0', 'start_frame': '134', 'end_frame': '214'},
 {'label': '0', 'start_frame': '132', 'end_frame': '212'},
{'label': '0', 'start_frame': '125', 'end_frame': '205'},
{'label': '0', 'start_frame': '152', 'end_frame': '232'},
{'label': '0', 'start_frame': '142', 'end_frame': '222'},
{'label': '0', 'start_frame': '140', 'end_frame': '220'},
{'label': '0', 'start_frame': '134', 'end_frame': '214'},
{'label': '0', 'start_frame': '160', 'end_frame': '240'},
{'label': '0', 'start_frame': '140', 'end_frame': '220'},
{'label': '0', 'start_frame': '137', 'end_frame': '217'},
{'label': '0', 'start_frame': '142', 'end_frame': '222'},
{'label': '0', 'start_frame': '135', 'end_frame': '215'},
{'label': '0', 'start_frame': '127', 'end_frame': '207'},
{'label': '0', 'start_frame': '131', 'end_frame': '211'},
{'label': '0', 'start_frame': '131', 'end_frame': '211'},
{'label': '0', 'start_frame': '131', 'end_frame': '211'},
{'label': '0', 'start_frame': '119', 'end_frame': '199'},
{'label': '0', 'start_frame': '131', 'end_frame': '211'}]

val0Annotations=[
{"subset": "validation", "annotations": {"label": "0", "start_frame": "222", "end_frame": "302"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "143", "end_frame": "223"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "132", "end_frame": "212"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "98", "end_frame": "178"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "147", "end_frame": "227"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "128", "end_frame": "208"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "145", "end_frame": "225"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "112", "end_frame": "192"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "113", "end_frame": "193"}},
{"subset": "validation", "annotations": {"label": "0", "start_frame": "125", "end_frame": "205"}}]

val0List=[
'class0/01subject1_r0',
'class0/02subject12_r2',
'class0/03subject16_r2',
'class0/04subject20_r0',
'class0/05subject16_r5',
'class0/06subject6_r2',
'class0/07subject13_r0',
'class0/08subject14_r0',
'class0/09subject8_r0',
'class0/10subject12_r0'
]

datasetPath='./TrainImg'
dstJsonPath='./SFHTrain.json'
if __name__=='__main__':
    jsondata={}

    jsondata['labels']=['0','1']
    jsondata['database']={}
    class1Path= datasetPath+'/class1'
    for dir in os.listdir(class1Path):
        videopath=class1Path+'/'+dir
        endFrame=len(os.listdir(videopath))
        print(videopath,' end frame: ',endFrame)
        jsondata['database'][videopath]={'subset':'training','annotations':{'label':'1','start_frame':'1','end_frame':endFrame}}
        print(jsondata['database'][videopath])

    for i in range(len(class0List)):
        videopath = datasetPath + '/' + class0List[i]
        jsondata['database'][videopath]={'subset':'training','annotations':class0Annotations[i]}

        #################VALIDATION SET####################
    datasetPath='./ValImg'
    class1Path = datasetPath + '/class1'
    for dir in os.listdir(class1Path):
        videopath = class1Path + '/' + dir
        endFrame = len(os.listdir(videopath))
        print(videopath, ' end frame: ', endFrame)
        jsondata['database'][videopath] = {'subset': 'validation',
                                           'annotations': {'label': '1', 'start_frame': '1', 'end_frame': endFrame}}

    for i in range(len(val0List)):
        videopath = datasetPath + '/' + val0List[i]
        jsondata['database'][videopath] = val0Annotations[i]

    with open(dstJsonPath, 'w') as dst_file:
        json.dump(jsondata, dst_file)
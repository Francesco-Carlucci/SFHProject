import os
from subprocess import call


datasetPath='./class0'
dstPath='./class0Img'
if __name__=='__main__':
    i=1
    for file in os.listdir(datasetPath):

        dstDir=os.path.join(dstPath,'%03d'%i)
        i+=1
        print('extracting frames of: ',file,' in ',dstDir)
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)
        directory = os.path.join(dstDir, "%05d.jpg").replace('\\', '/')
        file=datasetPath+"/"+file
        call(["ffmpeg", "-i", file, directory, "-hide_banner"])
import os
import face_recognition
root_path="/home/hp/PycharmProjects/untitled/person/"
files=os.listdir(root_path)
facelist=[]
namelist=[]
def get_known():
    for file in files:
        img=face_recognition.load_image_file(root_path + file)
        FE=face_recognition.face_encodings(img)[0]
        facelist.append(FE)
        name=file.split('_')
        namelist.append(name[0])
    return facelist,namelist

import asyncio
import websockets
import base64 
from PIL import Image
import numpy as np
import cv2
from Emotion_recognition_master import emotion_detection
from Keras_age_gender_master import age_gender_detection
import ML_MODEL


threshold_accuracy_of_person=0.3

background_status=""

NO_OF_SAMPLE=0

pridictionEnable=0

min_sample_to_start_pridiction=100


all_peolpe_nose_arrX=[]
all_peolpe_nose_arrY=[]

DATASETSTRING=""


def prediction_of_emotion(DATASETSTRING,image):
    testX=[]
    actualemotion=[]
    pridictionarr=[]
    arr=DATASETSTRING.split("\n")
    for i in arr:
        arr1=i.split("\t")
        row=[]
        if len(arr1)>=48:
            for j in range(0,len(arr1)-1):
                if j!=36:
                    row.append(int(arr1[j]))
            testX.append(row)
            actualemotion.append(int(arr1[36]))
    
    for j in testX:
        pridictionarr.append(ML_MODEL.test(j))

    print("actualemotion",actualemotion)
    print("pridictionarr",pridictionarr)
    


    for h in range(0,len(testX)):
        if pridictionarr[h]=="unpredictible":
            image = cv2.circle(image,(testX[h][0],testX[h][1]),50,(255,0,0),3)
        elif pridictionarr[h]!=actualemotion[h]:
            image = cv2.circle(image,(testX[h][0],testX[h][1]),50,(0,0,255),3) 
        else:
            image = cv2.circle(image,(testX[h][0],testX[h][1]),50,(0,255,0),3) 
 

    return(image)

            


        


        







def reletionship(skeleton_and_face_gea):
    global all_peolpe_nose_arrX
    global all_peolpe_nose_arrY
    global DATASETSTRING

    if len(all_peolpe_nose_arrX)>1 and len(all_peolpe_nose_arrY)>1:

        arrDATASET=DATASETSTRING.split("\n")
        arrDATASET=arrDATASET[1:]
        

        for i in range(0,len(all_peolpe_nose_arrX)):
            arrDATASET[i]=arrDATASET[i]+str(len(all_peolpe_nose_arrX)-1)
            arrDATASET[i]=arrDATASET[i]+"\t"
            for j in range(0,len(all_peolpe_nose_arrX)):
                if i!=j:
                    if all_peolpe_nose_arrX[i]<all_peolpe_nose_arrX[j]:
                        firstx=all_peolpe_nose_arrX[i]
                        secondx=all_peolpe_nose_arrX[j]
                        firsty=all_peolpe_nose_arrY[i]
                        secondy=all_peolpe_nose_arrY[j]
                    else:
                        firstx=all_peolpe_nose_arrX[j]
                        secondx=all_peolpe_nose_arrX[i]
                        firsty=all_peolpe_nose_arrY[j]
                        secondy=all_peolpe_nose_arrY[i]
                    arrDATASET[i]=arrDATASET[i]+str(secondx-firstx)
                    arrDATASET[i]=arrDATASET[i]+"\t"               
                    cv2.line(skeleton_and_face_gea,(firstx,firsty),(secondx,secondy),(111, 38, 255),6)
            for m in range(len(all_peolpe_nose_arrX)-1,10):
                 arrDATASET[i]=arrDATASET[i]+"-1"
                 arrDATASET[i]=arrDATASET[i]+"\t"  

        DATASETSTRING=""
        DATASETSTRING=DATASETSTRING+"\n"
        for f in arrDATASET:
            DATASETSTRING=DATASETSTRING+f
            DATASETSTRING=DATASETSTRING+"\n"

    else:
        DATASETSTRING=DATASETSTRING+"0"
        DATASETSTRING=DATASETSTRING+"\t"
        for i in range(0,10):
            DATASETSTRING=DATASETSTRING+"-1"
            DATASETSTRING=DATASETSTRING+"\t"


    
    skeleton_and_face_gea_relations=skeleton_and_face_gea
    return(skeleton_and_face_gea_relations)



def Drow_age_gender_emotion(skeleton_and_face,leftear,rightear,leftsholder,rightsholder):
    try:
        agegender=age_gender_detection.age_gender_detect()
        arr=agegender.split("/")
        gender=arr[0]
        age=arr[1]
        emotion=emotion_detection.emotion_detect()
        cv2.putText(skeleton_and_face,gender+" "+age+" "+emotion,(rightear[0],rightear[1]-50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        skeleton_and_face_gea=skeleton_and_face


        global DATASETSTRING
        DATASETSTRING=DATASETSTRING+age
        DATASETSTRING=DATASETSTRING+"\t"


        if gender=="M":
            DATASETSTRING=DATASETSTRING+"11"
            DATASETSTRING=DATASETSTRING+"\t"
        else:
            DATASETSTRING=DATASETSTRING+"99"
            DATASETSTRING=DATASETSTRING+"\t"


        #["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
        if emotion=="angry":
            DATASETSTRING=DATASETSTRING+"1000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="disgust":
            DATASETSTRING=DATASETSTRING+"2000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="scared":
            DATASETSTRING=DATASETSTRING+"3000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="happy":
            DATASETSTRING=DATASETSTRING+"4000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="sad":
            DATASETSTRING=DATASETSTRING+"5000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="surprised":
            DATASETSTRING=DATASETSTRING+"6000"
            DATASETSTRING=DATASETSTRING+"\t"
        if emotion=="neutral":
            DATASETSTRING=DATASETSTRING+"7000"
            DATASETSTRING=DATASETSTRING+"\t"
        

        return(skeleton_and_face_gea)


    except:
        print("UNABLE TO GET THE AGR GENDER AND EMOTION")
        return("UNKNOWN")



def save_face_drow_face(skeleton,leftear,rightear,leftsholder,rightsholder):

    image = cv2.imread("stream.png")
    temp=rightsholder[1]-rightear[1]
    cimage=image[rightear[1]-temp:rightsholder[1],rightsholder[0]:leftsholder[0]]
    h,w,d=cimage.shape
    if h>0 and w>0:
        cv2.imwrite("face.png",cimage)
    else:
        print("unable to crop properly")

        


    color = (255,0,255) 
    thickness = 5

    cv2.rectangle(skeleton,(rightsholder[0],rightear[1]-temp),(leftsholder[0],leftsholder[1]), color, thickness)

    skeleton_and_face=skeleton


    return(skeleton_and_face)



def Drow_skeleton(image,arr1):
    global DATASETSTRING
    global all_peolpe_nose_arrX
    global all_peolpe_nose_arrY

    color = (255, 255, 0) 
    thickness = 5

    

    #nose-lefteye
    cv2.line(image,(int(float(arr1[1])),int(float(arr1[2]))),(int(float(arr1[4])),int(float(arr1[5]))), color, thickness)
    #nose-righteye
    cv2.line(image,(int(float(arr1[1])),int(float(arr1[2]))),(int(float(arr1[7])),int(float(arr1[8]))), color, thickness)
    #lefteye-leftear
    cv2.line(image,(int(float(arr1[4])),int(float(arr1[5]))),(int(float(arr1[10])),int(float(arr1[11]))), color, thickness)
    #righteye-rightear
    cv2.line(image,(int(float(arr1[7])),int(float(arr1[8]))),(int(float(arr1[13])),int(float(arr1[14]))), color, thickness)
    #nose-leftsholder
    cv2.line(image,(int(float(arr1[1])),int(float(arr1[2]))),(int(float(arr1[16])),int(float(arr1[17]))), color, thickness)
    #nose-rightsholder
    cv2.line(image,(int(float(arr1[1])),int(float(arr1[2]))),(int(float(arr1[19])),int(float(arr1[20]))), color, thickness)
    #leftsholder-leftelbo 
    cv2.line(image,(int(float(arr1[16])),int(float(arr1[17]))),(int(float(arr1[22])),int(float(arr1[23]))), color, thickness)
    #rigthsholder-rigthelbo
    cv2.line(image,(int(float(arr1[19])),int(float(arr1[20]))),(int(float(arr1[25])),int(float(arr1[26]))), color, thickness)  
    #leftelbo-leftwrist
    cv2.line(image,(int(float(arr1[22])),int(float(arr1[23]))),(int(float(arr1[28])),int(float(arr1[29]))), color, thickness)
    #rightelbo-rigthwrist
    cv2.line(image,(int(float(arr1[25])),int(float(arr1[26]))),(int(float(arr1[31])),int(float(arr1[32]))), color, thickness)
    #leftsholder-leftheap
    cv2.line(image,(int(float(arr1[16])),int(float(arr1[17]))),(int(float(arr1[34])),int(float(arr1[35]))), color, thickness)
    #rightsholder-rigtheap
    cv2.line(image,(int(float(arr1[19])),int(float(arr1[20]))),(int(float(arr1[37])),int(float(arr1[38]))), color, thickness)
    #leftheap-leftknee
    cv2.line(image,(int(float(arr1[34])),int(float(arr1[35]))),(int(float(arr1[40])),int(float(arr1[41]))), color, thickness)
    #rightheap-rightknee
    cv2.line(image,(int(float(arr1[37])),int(float(arr1[38]))),(int(float(arr1[43])),int(float(arr1[44]))), color, thickness)
    #leftknee-leftankle
    cv2.line(image,(int(float(arr1[40])),int(float(arr1[41]))),(int(float(arr1[46])),int(float(arr1[47]))), color, thickness)  
    #rightknee-rightankle
    cv2.line(image,(int(float(arr1[43])),int(float(arr1[44]))),(int(float(arr1[49])),int(float(arr1[50]))), color, thickness)

    skeleton=image
    

    all_peolpe_nose_arrX.append(int(float(arr1[1])))   
    all_peolpe_nose_arrY.append(int(float(arr1[2]))) 


    
    for i in range(1,51):
        if i%3!=0:
            DATASETSTRING=DATASETSTRING+str(int(float(arr1[i])))
            DATASETSTRING=DATASETSTRING+"\t"


    return(skeleton)


def processing_image(posedata):
    global background_status
    global DATASETSTRING
    global NO_OF_SAMPLE
    print("background_status")
    print(background_status)
    if background_status=="1":
        image = cv2.imread("stream.png")
    else:
        image = cv2.imread("background.png")



    arr1=posedata.split("%")
    no_of_person=arr1[0]
    arr2=posedata.split("????")
    print("no_of_person: "+no_of_person)
    for i in range(0,int(no_of_person)):
        arr3=arr2[i].split("$")
        arr4=arr3[0].split("||")
        accuracy=arr4[1]
        print("person_no: "+str(i+1))
        print("accuracy: "+accuracy)

        if float(accuracy)>threshold_accuracy_of_person:

            NO_OF_SAMPLE=NO_OF_SAMPLE+1
            DATASETSTRING=DATASETSTRING+"\n"

            arr4=arr3[1].split("&")   
            skeleton=Drow_skeleton(image,arr4)
            skeleton_and_face=save_face_drow_face(skeleton,[int(float(arr4[10])),int(float(arr4[11]))],[int(float(arr4[13])),int(float(arr4[14]))],[int(float(arr4[16])),int(float(arr4[17]))],[int(float(arr4[19])),int(float(arr4[20]))])
            skeleton_and_face_gea=Drow_age_gender_emotion(skeleton_and_face,[int(float(arr4[10])),int(float(arr4[11]))],[int(float(arr4[13])),int(float(arr4[14]))],[int(float(arr4[16])),int(float(arr4[17]))],[int(float(arr4[19])),int(float(arr4[20]))])
            processed_image=skeleton_and_face_gea
        else:
            processed_image=image


    global all_peolpe_nose_arrX
    global all_peolpe_nose_arrY


    if len(all_peolpe_nose_arrX)>=1 and len(all_peolpe_nose_arrY)>=1:
        skeleton_and_face_gea_relations=reletionship(processed_image)
        processed_image=skeleton_and_face_gea_relations

    all_peolpe_nose_arrX.clear()
    all_peolpe_nose_arrY.clear()


    global pridictionEnable
    if pridictionEnable==1:
        skeleton_and_face_gea_relations_prediction=prediction_of_emotion(DATASETSTRING,processed_image)
        processed_image=skeleton_and_face_gea_relations_prediction


    FILE=open("Dataset.txt","a")
    FILE.write(DATASETSTRING)
    DATASETSTRING=""
    FILE.close()
    print("SAVED")

    cv2.imwrite("finalimage.png",processed_image)


    with open("finalimage.png","rb") as image_file:
        baseimg=base64.b64encode(image_file.read())
    baseimg=str(baseimg)
    arrf=baseimg.split("'")
    finalpic="data:image/png;base64,"+arrf[1]
    return(finalpic)


        










def imgsave(fname,base64str):
    img=base64str.split(",")
    pic=base64.b64decode(img[1])
    f=open(fname,"wb")
    f.write(pic)
    f.close()  
 














async def time(websocket, path):
    print(websocket)


    global pridictionEnable
    global NO_OF_SAMPLE
    global min_sample_to_start_pridiction
    print("NO_OF_SAMPLE==",NO_OF_SAMPLE)
    if NO_OF_SAMPLE>min_sample_to_start_pridiction:
        ML_MODEL.train()
        NO_OF_SAMPLE=0
        pridictionEnable=1

    data= await websocket.recv()
    if len(data)>0:
        arr=data.split("::::")
        global background_status
        background_status=arr[0]
        arrdata=arr[1].split("**********")
        posedata=arrdata[0]
        base64str = arrdata[1]

        imgname="stream.png"
        imgsave(imgname,base64str)

 
        fianalpic=processing_image(posedata)
        await websocket.send(fianalpic)
        print("SEND")











emotion_detection.initialize_emotion_detection_model_loading_process()
age_gender_detection.initialize_age_gender_detection_model_loading_process()
print("emotion_detection and age_gender_detection INITIALIZED")

FILE=open("Dataset.txt","w")
FILE.write("")
FILE.close()
print("Dataset.txt file cleaned")


print("server is online")
start_server = websockets.serve(time, '192.168.43.7', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy

neigh=None
tc=None
LR=None
LDA=None

throushed=3

def train():
    global neigh
    global tc
    global LR
    global LDA
    trainarrX=[]
    trainarrY=[]
    file = open("Dataset.txt","r")
    for i in file:
        arr1=i.split("\t")
        row=[]
        if len(arr1)>=48:
            for j in range(0,len(arr1)-1):
                if j!=36:
                    row.append(int(arr1[j]))
            trainarrX.append(row)
            trainarrY.append(int(arr1[36]))


    trainarrX = numpy.array(trainarrX, dtype=numpy.int32)
    trainarrY = numpy.array(trainarrY, dtype=numpy.int32)


    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(trainarrX,trainarrY)

    tc = tree.DecisionTreeClassifier()
    tc.fit(trainarrX,trainarrY)

    LR= LogisticRegression(random_state=0)
    LR.fit(trainarrX,trainarrY)

    LDA= LinearDiscriminantAnalysis()
    LDA.fit(trainarrX,trainarrY)

    print("**************************ALL MODELS HAS BEEN TRAINED************************")




def test(testdata):

    global neigh
    global tc
    global LR
    global LDA
    global throushed

    testX=[testdata]

    testX = numpy.array(testX, dtype=numpy.int32)

    output=[]

    output.append(neigh.predict(testX)[0])

    output.append(tc.predict(testX)[0])

    output.append(LR.predict(testX)[0])

    output.append(LDA.predict(testX)[0])

    print(output)

    count=[0,0,0,0,0,0,0]
    for i in output:
        if i==1000:
            count[0]=count[0]+1
        if i==2000:
            count[1]=count[1]+1
        if i==3000:
            count[2]=count[2]+1
        if i==4000:
            count[3]=count[3]+1
        if i==5000:
            count[4]=count[4]+1
        if i==6000:
            count[5]=count[5]+1
        if i==7000:
            count[6]=count[6]+1


    

    emotionnumberarr=[1000,2000,3000,4000,5000,6000,7000]
    if max(count)>=throushed:
        index=count.index(max(count))
        emotionnumber=emotionnumberarr[index] 
        return(emotionnumber)
    else:
        return("unpredictible")




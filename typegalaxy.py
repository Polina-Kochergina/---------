import csv
import numpy as np
import os
from urllib.request import urlretrieve
from PIL import Image, ImageDraw
# import random

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def make_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=3, activation='relu',
    input_shape=(32, 32, 1)))
    model.add(Conv2D(4, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
    data_format=None))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
    data_format=None))
    model.add(Flatten())
    model.add(Dropout(rate=0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model

data_EL = []
data_EDGE = []
data_ACW = []
data_CW = []

# print(my_data[0])
with open('Galaxy.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(len(row))
        if (float(row['P_EL']) > 0.8):
            data_EL.append([row['RA'],row['DEC'], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])       
        elif (float(row['P_EDGE']) > 0.8):
            data_EDGE.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])
        elif (float(row['P_ACW']) > 0.8):
            data_ACW.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])
        elif (float(row['P_CW']) > 0.8):
            data_CW.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']]) 


def files_writer(ads, name):
    with open(f'parsed_{name}.csv', 'w', encoding='utf-8') as file:
        a_pen = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_NONE, quotechar=' ', escapechar=' ', lineterminator = '\n')
        a_pen.writerow(('RA', 'DEC'))
        for ad in ads:
            # print(ad[0])
            a_pen.writerow(('nearest:B',ad[0], ad[1], ad[2], ad[3], ad[4], ad[5]))
            
# сохранение файлов
files_writer(data_EL, 'EL')
files_writer(data_EDGE, 'EDGE')
files_writer(data_ACW, 'ACW')
files_writer(data_CW, 'CW')

print('save file')

list = os.listdir("logd25")
def Read_RA_DEC(name, dir, file):
    print(name, file)
    # for i in range(len(list)):
    with open(f'{dir}\{file}') as csvfile:
        if (os.path.exists(f"image_{name}") == False):
            os.mkdir(f"image_{name}")
        reader = csv.reader(csvfile, delimiter='|')
        listCW = []
        p = open
        for key in reader:
            if ((key[0][0] != '!')and(key[4].strip() != '') and (10**float(key[4])*6 > 15)):
                # print(key[0])
                # print(key[1])
                # print(key[0])
                line = key[0].lstrip("nearest:B ")
                lineRA, lineDEC = line.split()
                h, m, s = lineRA.split(":")
                h = float(h); m = float(m); s = float(s)
                d, min, sec = lineDEC.split(":")
                d = float(d); min = float(min); sec = float(sec)

                lineRA = (h*15*3600 + m*15*60 + s*15)/3600
                lineDEC = (abs(d*3600) + min*60 + sec)/3600
                if int(d)<0:
                    lineDEC = lineDEC*(-1)
                line = [lineRA, lineDEC,float(key[4].strip()), key[1].strip()]

                # line.append(float(key[4].strip()))
                listCW.append(line)
    print(len(listCW))
#     используем url для доступа к серверу
    for item in range(len(listCW)):
        if (item < 1000):
            scale = listCW[item][2]/1.5
            site = f"https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={listCW[item][0]}&dec={listCW[item][1]}&scale={scale}&width=64&height=64"
            urlretrieve(site, f"image_{name}\image_{listCW[item][3]}.png")
        
    print(name)
           
# Read_RA_DEC("ACW", 'logd25' , list[1])
# Read_RA_DEC("CW", 'logd25', list[2])
# Read_RA_DEC("ALL", 'logd25', list[0])
# Read_RA_DEC("EDGE", 'logd25', list[3])
# Read_RA_DEC("El", 'logd25', list[4])

# print(len(listCW))
# print(listCW[5][0])
# print(listCW[5][1])
# print(len(reader))

# используем url для доступа к серверу
# site = f"http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={args.ra}&dec={args.dec}&scale=0.5&width={args.w}&height={args.h}&opt=G"
# urlretrieve(site, f"{args.name}.jpeg")



# Split sample on training and test subsamples
def open_norm(dir):
    list = os.listdir(dir)
    path = os.getcwd()
    print(path)
    os.chdir(f'{dir}')
    for file in list:
        # print(file)
        path = os.getcwd()
        # print(path)
        path = os.path.abspath(file)
        # print(path)
        with Image.open(path) as img:
            # print(img[0][0])
            arr = np.asarray(img)
            # print(arr)
            # print(arr[0][0])

            max = arr.max()
            # print(max)
            arr = arr/max
            # print('fkhgj')
            # arr.save(f'{dir}\{file}')
            # if (os.path.exists(f"image_{name}") == False):
            #     os.mkdir(f"image_{name}")
        # reader = csv.reader(csvfile, delimiter='|')
        # listCW = []

# print("close dir")

# open_norm('image_ACW')
# open_norm('image_CW')
# open_norm('image_EDGE')
# open_norm('image_EL')
img_size = 64

data_train, data_test, labels_train, labels_test =train_test_split(data, labels, test_size=0.20)
data_train = data_train.reshape(-1, img_size, img_size, 1)
data_test = data_test.reshape(-1, img_size, img_size, 1)
model = make_model()
model.fit(data_train, labels_train,
validation_data=(data_test, labels_test),
epochs=5, batch_size=32)

print('DONE!')
# print(data_EL[0][1])
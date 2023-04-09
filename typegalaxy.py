import csv
import numpy as np
import os
from urllib.request import urlretrieve
# from numpy import genfromtxt

data_EL = []
data_EDGE = []
data_ACW = []
data_CW = []

# print(my_data[0])
# with open('Galaxy.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         # print(len(row))
#         if (float(row['P_EL']) > 0.8):
#             data_EL.append([row['RA'],row['DEC'], row['P_EL']])       
#         elif (float(row['P_EDGE']) > 0.8):
#             data_EDGE.append([row["RA"],row["DEC"], row['P_EDGE']])
#         elif (float(row['P_ACW']) > 0.8):
#             data_ACW.append([row["RA"],row["DEC"], row['P_ACW']])
#         elif (float(row['P_CW']) > 0.8):
#             data_CW.append([row["RA"],row["DEC"], row['P_CW']]) 


def files_writer(ads, name):
    with open(f'parsed_{name}.csv', 'w', encoding='utf-8') as file:
        a_pen = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_NONE, quotechar=' ', escapechar=' ', lineterminator = '\n')
        a_pen.writerow(('RA', 'DEC'))
        for ad in ads:
            # print(ad[0])
            a_pen.writerow(('nearest:B',ad[0], ad[1], ad[2]))
            
# сохранение файлов
# files_writer(data_EL, 'EL')
# files_writer(data_EDGE, 'EDGE')
# files_writer(data_ACW, 'ACW')
# files_writer(data_CW, 'CW')

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
        for key in reader:
            if ((key[0][0] != '!')and(key[1].strip() != '') and (10**float(key[1])*6 > 15)):
                # print(key[0])
                # print(key[1])
                # print(key[0])
                line = key[0].lstrip("nearest:B ")
                h, m, s, d, min, sec = line.split()
                # h, m, s = lineRA.split(":")
                h = float(h); m = float(m); s = float(s)
                # d, min, sec = lineDEC.split(":")
                d = float(d); min = float(min); sec = float(sec)

                lineRA = (h*15*3600 + m*15*60 + s*15)/3600
                lineDEC = (abs(d*3600) + min*60 + sec)/3600
                if int(d)<0:
                    lineDEC = lineDEC*(-1)
                line = [lineRA, lineDEC,float(key[1].strip())]

                # line.append(float(key[4].strip()))
                listCW.append(line)
    print(len(listCW))
#     используем url для доступа к серверу
    for item in range(len(listCW)):
        if (item < 5000)and(item >3650):
            scale = listCW[item][2]/1.5
            site = f"https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={listCW[item][0]}&dec={listCW[item][1]}&scale={scale}&width=64&height=64"
            urlretrieve(site, f"image_{name}\image_{item}.jpeg")
        
            # listACW.append([key[0],key[4]])
            # print(key[2])
            # else:
                # row[]
# Read_RA_DEC("ACW", 'logd25' , list[1])
# Read_RA_DEC("CW", 'logd25', list[2])
Read_RA_DEC("ALL", 'logd25', list[0])
# Read_RA_DEC("EDGE", 'logd25', list[3])
# Read_RA_DEC("El", 'logd25', list[4])

# print(len(listCW))
# print(listCW[5][0])
# print(listCW[5][1])
# print(len(reader))

# используем url для доступа к серверу
# site = f"http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={args.ra}&dec={args.dec}&scale=0.5&width={args.w}&height={args.h}&opt=G"
# urlretrieve(site, f"{args.name}.jpeg")

print('DONE!')
# print(data_EL[0][1])
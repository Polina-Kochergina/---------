import csv
from matplotlib import pyplot as plt
import numpy as np
import os
from urllib.request import urlretrieve
from PIL import Image
from astropy.io import fits


from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from numpy import loadtxt
from keras.models import load_model

# это функция из презы --- модель сетки
def make_model():
    img_shape = (64, 64, 1)

    model = Sequential()
    model.add(Conv2D (4, kernel_size = 3, \
            activation = 'relu', input_shape = img_shape))
    model.add(Conv2D(4, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides= None, \
            padding='valid', data_format = None))
    model.add(Dropout(rate = 0.3))
    model.add(Conv2D(8, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(8, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = None,\
            padding = 'valid', data_format = None))
    model.add(Flatten())
    model.add(Dropout(rate = 0.3))
    model.add(Dense(50, activation = "relu"))
    model.add(Dense(4, activation = "softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy",\
            metrics=['accuracy'])

    return model



# тут я открываю данный файл от савченко и вытаскиваю те галактики, которые удовлетворяют условию что у них вероятность больше 0.8 (чтобы галактики получше отобрать)


# data_EL = []
# data_EDGE = []
# data_ACW = []
# data_CW = []

# # print(my_data[0])
# with open('Galaxy.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         # print(len(row))
#         if (float(row['P_EL']) > 0.8):
#             data_EL.append([row['RA'],row['DEC'], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])       
#         elif (float(row['P_EDGE']) > 0.8):
#             data_EDGE.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])
#         elif (float(row['P_ACW']) > 0.8):
#             data_ACW.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']])
#         elif (float(row['P_CW']) > 0.8):
#             data_CW.append([row["RA"],row["DEC"], row['P_EL'], row['P_EDGE'], row['P_ACW'], row['P_CW']]) 

# функция, которая создает csv файл для сайта в формате "nearest B: ...""
def files_writer(ads, name):
    with open(f'parsed_{name}.csv', 'w', encoding='utf-8') as file:
        a_pen = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_NONE, quotechar=' ', escapechar=' ', lineterminator = '\n')
        a_pen.writerow(('RA', 'DEC'))
        for ad in ads:
            # print(ad[0])
            a_pen.writerow(('nearest:B',ad[0], ad[1], ad[2], ad[3], ad[4], ad[5]))
            
# # сохранение файлов
# files_writer(data_EL, 'EL')
# files_writer(data_EDGE, 'EDGE')
# files_writer(data_ACW, 'ACW')
# files_writer(data_CW, 'CW')
# print('save file')

# на этом этапе, полученные файлы отправила на скормление сайту, он вернул мне другие файлы вида uploader_*.cgi


list = os.listdir("logd25")

# эта функция открывает *.cgi и переводит в радианы из часов минут секунд и тд
# (это нужно для сайта с которого качать картинки, он принимает ra и dec в радианах)
def Read_RA_DEC(name, dir, file):

    # print(name, file)
    # открываю uploader_*.cgi
    with open(f'{dir}\{file}') as csvfile:

        # для каждого типа галактик создаю папку image_{тип галактик}
        if (os.path.exists(f"image_{name}") == False):
            os.mkdir(f"image_{name}")

        reader = csv.reader(csvfile, delimiter='|') #читаем cgi как csv :)))))
        listCW = []

        for key in reader:

            # выбираем только те строки которые не закоменчены (!) где есть logd25 и такой чтобы он был > 15 секунд,
            #  но вот это кажется не оч нормальный перевод: 10**float(key[4])*6 > 15 --- неправильно
            if ((key[0][0] != '!')and(key[4].strip() != '') and (10**float(key[4])*6 > 15)):
                
                # тут начинается перевод в радианы
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

                # в переменную line записываю ra, dec, logd25, имя галактики
                line = [lineRA, lineDEC,float(key[4].strip()), key[1].strip()]

                # и все это добавляю к большому листу всех галактик этого типа
                listCW.append(line)


    # используем url для доступа к серверу
    # скачиваем картинки используя listCW (он так называется потому что изначально я видимо писала для спиральных по часовой стрелке,
    # но в принципе он для каждого типа свой просто назван не оч корректно)
    for item in range(len(listCW)):
        if (item > 1500) and (item < 1515):
            scale = listCW[item][2]/1.5
            site = f"https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={listCW[item][0]}&dec={listCW[item][1]}&scale={scale}&width=64&height=64"
            urlretrieve(site, f"image_{name}\image_{listCW[item][3]}.png")
           
# Read_RA_DEC("ACW", 'logd25' , list[1])
# Read_RA_DEC("CW", 'logd25', list[2])
# Read_RA_DEC("ALL", 'logd25', list[0])
# Read_RA_DEC("EDGE", 'logd25', list[3])
# Read_RA_DEC("El", 'logd25', list[4])

# эта функция не нужна здесь кажется, в общем это видимо была попытка какого-то костыля
def open_file(dir):
    filename = os.listdir(dir)
    hdu0 = fits.open(os.path.join(dir, filename[0]))
    # file0 = np.array(hdu0[0].data)
    hdu1 = fits.open(os.path.join(dir, filename[1]))
    # file1 = np.array(hdu1[0].data)



    data = np.array([[]])
    for i in range(len(hdu0)):
        data = np.append(data, hdu0[i].data)

    labels = np.array([[]])
    for i in range(len(hdu1)):
        labels = np.append(labels, hdu1[i].data)

    return data, labels 

# Split sample on training and test subsamples
# эта функция нормирует картинки и перезаписывает   ---ЭТО ВСЕ ЛОЖЬ эта функция тут не используется, точнее ее можно юзать правда я не написала что она возвращает и не прописываю ей перезаписать)
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
            arr = np.asarray(img)
            max = arr.max()
            arr = arr/max
            

# а это функция делает страшное (она засовывает в data и label по num(у меня 960) картинок из каждой папки с картинками
# и сует это все в 1 конечный массив, который будет подаваться сетке, и тоже самое с вероятностями)
def h(dir, num):
    data = np.empty((4*3*num, 64, 64))
    labels = np.zeros((4*3*num, 4))
    name = []
    radec = []
    imdir = os.listdir(dir)
    for k in range(len(imdir)):
        lst = os.listdir(f"{dir}\{imdir[k]}")
        print("длина list", len(lst))

        for item, file in enumerate(lst):
            # print(item)
            while item < num :  
                n = file.rstrip(".png")
                name.append(n.lstrip("image_"))
                # тут открывается png 
                with Image.open(f"{dir}\{imdir[k]}\{file}") as img:
                    arr = np.asarray(img)
                    # сумма т.к он  rgb(т.е три слоя, а нужен 1) я "перевела" в чб
                    arr = np.sum(arr, 2)
                    #  это отражение 
                    arr1 = np.fliplr(arr)
                    # а это поворот для увеличения выборки
                    arr2 = np.rot90(arr1)


                data[3*num*k + 3*item] = arr
                data[3*num*k + 3*item + 1] = arr1
                data[3*num*k + 3*item + 2] = arr2

                break
    # print(data.shape, "datash1")
    data = data.reshape(-1, 64, 64)
    # name1 = set(name)

    print(data.shape)
    cgi = os.listdir('logd25')
    pars = os.listdir('csv')
    print(pars)
    print(cgi)


    print('записали data ')
    # new_name = []
    # это момент когда по названию галактики находяся его ра и дек а потом по ра и дек из другого файла найдутся вероятности()()() оч плохо и костыльно 
    for k in range(1,len(cgi)):
        find_kolvo = 0
        for i in range(num):

            with open(f'logd25\{cgi[k]}') as csvfile:
                reader = csv.reader(csvfile, delimiter='|')
                # print(reader.dialect)
                for key in reader:
                    
                    if (key[0][0] != '!')and(key[1].strip() == name[(k-1)*num+i]):

                        line = key[0].lstrip("nearest:B ")
                        lineRA, lineDEC = line.split()
                        radec.append([name[(k-1)*num+i], lineRA, lineDEC])
                        # new_name.append(key[1].strip())
                        find_kolvo += 1
                    
                        break


        print(find_kolvo, "galaxies from", imdir[k-1])
    



    
    print('записали ra и dec ')
    print(len(radec), "radec")


    # вот тут из файлов который дал савченко(?) кажется достаем вероятности
    for k in range(len(pars)):
        for i in range(num):

                with open(f'csv\{pars[k]}') as csvfile1:
                    reader = csv.reader(csvfile1, delimiter=' ')
                    for key in reader: 

                        if (radec[ k*num + i ][2] == key[2]):
                            
                            p = [0, 0, 0, 0]
                            row = [float(key[3]), float(key[4]), float(key[5]), float(key[6])]
                            max_var = max(row)
                            max_idx = row.index(max_var)
                            # p[max_idx] = 1
                           
                            labels[3*num*k + 3*i][max_idx] = 1
                            labels[3*num*k + 3*i + 1][max_idx] = 1
                            labels[3*num*k + 3*i + 2][max_idx] = 1

                            break

    print('записали вероятности ')

    # print(data.shape, ' new data shape add flprl')

    return data, np.array(labels)
# финишная прямая это data и labels, которые будут использоваться для сетки
data , labels = h("image", 960)  

# open_norm('image_ACW')
# open_norm('image_CW')
# open_norm('image_EDGE')
# open_norm('image_EL')
# размер изображения
img_size = 64
# из  презы astroimg  в информации
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.20)

data_train = data_train.reshape(-1, img_size, img_size, 1)
data_test = data_test.reshape(-1, img_size, img_size, 1)

model = make_model()
model.fit(data_train, labels_train, validation_data = (data_test, labels_test), epochs = 10, batch_size = 32)
# сохраняем модель
model.save("model.h5")

# загружаем модель
model = load_model('model.h5')

# даем сетке картинку на которой она не обучалась
with Image.open('''image_CW\image_PGC1431757.png''') as file:
    Xnew = np.asarray(file)
    print('shape1 x:', Xnew.shape)
    Xnew = np.sum(Xnew, 2)
    Xnew = Xnew.reshape(-1, 64, 64, 1)
    print('shape2 x:', Xnew.shape)

# получаем вероятности что она поподает в тот или другой тип: el, edge, ACW, CW
ynew = model.predict(Xnew)
print(ynew)
# ура! это все
# что с гитом..
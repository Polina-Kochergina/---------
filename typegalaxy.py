import csv
import numpy as np
import os
import cgi
# from numpy import genfromtxt

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
            data_EL.append([row['RA'],row['DEC']])       
        elif (float(row['P_EDGE']) > 0.8):
            data_EDGE.append([row["RA"],row["DEC"]])
        elif (float(row['P_ACW']) > 0.8):
            data_ACW.append([row["RA"],row["DEC"]])
        elif (float(row['P_CW']) > 0.8):
            data_CW.append([row["RA"],row["DEC"]]) 


def files_writer(ads, name):
    with open(f'parsed_{name}.csv', 'w', encoding='utf-8') as file:
        a_pen = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_NONE, quotechar=' ', escapechar=' ', lineterminator = '\n')
        a_pen.writerow(('RA', 'DEC'))
        for ad in ads:
            # print(ad[0])
            a_pen.writerow(('nearest:B',ad[0], ad[1]))
            

# files_writer(data_EL, 'EL')
# files_writer(data_EDGE, 'EDGE')
# files_writer(data_ACW, 'ACW')
# files_writer(data_CW, 'CW')
list = os.listdir('logd25')
listACW = []
for i in range(len(list)):
    with open(f'logd25\{list[i]}') as csvfile:
        # reader = cgi.FieldStorage(filetxt)
        reader = csv.reader(csvfile, delimiter='|')
        # del reader[['!! request processed by HyperLeda Tue Apr  4 23:13:07 2023'][0] == "!"]

        # reader = {key:reader[key] for key in reader if key['!! request processed by HyperLeda Tue Apr  4 23:13:07 2023'][0] == "!"}
        # print(reader[0][0])
        for key in reader:
            # print(key)
            if ((key[0][0] != '!')and(key[4].strip() != '')):
                line = key[0].lstrip("nearest:B ")
                line = line.split()
                # print()
                line.append(str(key[4]))
                listACW.append(line)
                continue
            # listACW.append([key[0],key[4]])
            # print(key[2])
            # else:
                # row[]
        
print(len(listACW))
print(listACW[5][0])
# print(len(reader))

print('DONE!')
# print(data_EL[0][1])
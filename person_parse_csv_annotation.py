import os
import csv
import cv2

def convertYoloToRegular(size, coord):

    x2 = int(((2*size[0]*float(coord[0]))+(size[0]*float(coord[2])))/2)
    x1 = int(((2*size[0]*float(coord[0]))-(size[0]*float(coord[2])))/2)

    y2 = int(((2*size[1]*float(coord[1]))+(size[1]*float(coord[3])))/2)
    y1 = int(((2*size[1]*float(coord[1]))-(size[1]*float(coord[3])))/2)
    print('output Regular-Labels from function: '+ str(x1) +' '+ str(y1) +' '+ str(x2) +' '+ str(y2))
    return (x1,y1,x2,y2)


#train_csv_file = open('person_train.csv', 'w')
csv_file = open('person_test_2.csv', 'w')

txt_file = open('/mnt/person/test_2.txt', 'r')
contents = txt_file.readlines()


for img_path in contents:
    img_path = img_path.strip('\n')
#img_path = contents[2].strip('\n')
    #print(img_path)

    img = cv2.imread(img_path)
    height, width, _ =  img.shape
    size = [width, height]
    #print(size)

    label_path = img_path.replace('jpg', 'txt').strip('\n')
    #print(label_path)

    label_txt_file = open(label_path, 'r')
    label_contents = label_txt_file.readlines()
    #print(label_contents[0].strip('\n')) 
    
    for label in label_contents:
        _list = label.strip('\n').split(' ')
        x, y, z, w = convertYoloToRegular(size, _list[1:])
        #print(x, y, z, w)
        #print(_list)
        _list[0] = img_path
        _list[1] = str(x)
        _list[2] = str(y)
        _list[3] = str(z)
        _list[4] = str(w)
        _list.append('person')
        print(_list)

        csv_file.write(','.join(_list) + '\n')

csv_file.close()

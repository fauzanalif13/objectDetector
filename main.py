#kita akan menggunakan mobilenet SSD utk object detection
#karena metode ini merupakan metode paling stabil antara cepat dan akurat
#bisa banyak object, dan real time

#kita tidak menggunakan yolo karena membutuhkan gpu
#tidak menggunakan yellow tiny krn bbrp object tdk bagus deteksinya
#kita jg menggunakan mobilenet ssd krn ini bisa digunakan d raspberry pi

import cv2

#img = cv2.imread('lena.png')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
threshVal = 0.6 #tingkat akurasi deteksi
color = (255,0,0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

classNames = []
classFiles = 'Object_Detection_Files\coco.names' #jumlah object yg dpt dideteksi adalah 91
with open(classFiles, 'rt') as f: #rt artinya read text
    classNames = f.read().rstrip('\n').split('\n')
print (classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

#ini adalah setting default
net = cv2.dnn_DetectionModel (weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5 , 127.5, 127.5))
net.setInputSwapRB(True)

#class ids adalah dimana kita memanggil file codename
#bbox adalah bounding box, utk detector
#confthreshold adalah detector, jika kurang dari 50%(0.5), maka tdk akan dideteksi

#kita perlu memasukkan ini ke while true
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=threshVal)
    print(classIds, bbox)

    #jadi if dibawah ini utk jika ada yg dideteksi, tp tdk empty
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color, 3)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                    font, 1.5, color, 2)
            cv2.putText(img, "AKURASI: " + str(int(confidence*100)) +"%", (box[0] + 10, box[1] + 50),
                        font, 1, color, 2)

    cv2.imshow("Video Image Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

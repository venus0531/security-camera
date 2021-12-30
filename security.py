import cv2
import sys
import requests
import datetime
import pytz

#画像の名前に利用する変数
save_count = 1
#スクショをする為のフラグ
picture_flag = 0
#検知した顔の数
face_number = 0
#xmlファイル
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml'

# Webカメラ
cap = cv2.VideoCapture(0)


#取得したLine Notify TOKEN
TOKEN = ''
#LINEAPIのURL
api_url = 'https://notify-api.line.me/api/notify'
#通知内容
send_messages = 'Security Warning!'
#認証情報
TOKEN_dic = {'Authorization' : 'Bearer' + ' ' + TOKEN}
#メッセージ情報
send_dic = {'message' : send_messages}


while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break

    #フレームをグレースケールに変換
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_file)
    detected_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30))
    print(detected_face_list)

    if len(detected_face_list):
        for (x,y,w,h) in detected_face_list:
            print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 200, 255), thickness=10)
            
        if face_number < len(detected_face_list):
            picture_flag = 1
            
        if face_number != len(detected_face_list):
            face_number = len(detected_face_list)
            
        if picture_flag == 1:
            cv2.imwrite("capture_{}.png".format(save_count), frame)
            
            #画像ファイルのパスを指定
            image_file = r'C:\Users\p-user\Desktop\capture_{}.png'.format(save_count)
            #バイナリーデータで読み込む
            binary = open(image_file, mode = 'rb')
            #指定の辞書型にする
            image_dic = {'imageFile' : binary}
            #LINEに画像とメッセージを送る
            requests.post(api_url, headers = TOKEN_dic, data = send_dic, files = image_dic)
            #フラグを0にする
            picture_flag = 0
            #save_countを1増やす
            save_count += 1

    else:
        print('not detected')
        face_number = 0
        picture_flag = 0

    cv2.imshow('image', frame)
    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


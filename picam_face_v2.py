import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import face_recognition

camera = PiCamera()
camera.rotation = 180
# 調整解析度
# camera.resolution = (640, 480)
camera.resolution = (320, 240)

# 調整FPS
# camera.framerate = 30
# camera.framerate = 15
camera.framerate = 5

rawCapture = PiRGBArray(camera)
time.sleep(0.1)

# 加載圖片
known_image = face_recognition.load_image_file("jcw.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# 取得攝像機畫面
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # 設定 OpenCV 格式圖像
    image = frame.array

    # 進行比對
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # 比對照片跟畫面
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        name = "Unknown"
        if matches[0]:
            name = "金城武"

        # 在畫面畫方框與判別
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # 顯示畫面
    cv2.imshow("Frame", image)

    # 清空緩存, 準備下一個
    rawCapture.truncate(0)

    # 按q 離開畫面
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # 加入延遲降低幀率
    time.sleep(0.2)

# 關閉窗口與攝像機
cv2.destroyAllWindows()
camera.close()

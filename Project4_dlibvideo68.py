import dlib
import cv2

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('68face.dat')

while(cap.isOpened()):
  ret, frame = cap.read()

  # 偵測人臉
  face_rects, scores, idx = detector.run(frame, 0)

  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = "%2.2f(%d)" % (scores[i], idx[i])
  
    # cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
    shape = landmark_predictor(frame,d)
    
    for i in range(68):
        cv2.circle(frame, (shape.part(i).x, shape.part(i).y),2,(0,255,0), -1, 8)
        # cv2.putText(frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))

  # 顯示結果
  cv2.imshow("Face Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
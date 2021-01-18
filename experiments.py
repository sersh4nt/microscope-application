import cv2

vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = vs.read()
cv2.imshow('img', frame)
cv2.waitKey(0)

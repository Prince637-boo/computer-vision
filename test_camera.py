import cv2

print("Ouverture de la caméra...")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera ouverte :", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERREUR: impossible de lire une frame")
        break

    cv2.imshow("TEST CAMERA", frame)

    # IMPORTANT
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Q detecte")
        break

cap.release()
cv2.destroyAllWindows()

print("Camera liberee.")
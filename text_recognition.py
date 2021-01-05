import easyocr
import cv2


def find_text(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 3)

    reader = easyocr.Reader(['en'])
    print(reader.readtext(image))
    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    find_text(cv2.imread("new.png"))

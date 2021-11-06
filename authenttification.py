import face_recognition
import cv2
import sys


def pren_photo():
    print("Scannnig face ....")
    cap = cv2.VideoCapture(0)
    ret , frame = cap.read()
    cv2.imwrite("photo.jpg",frame)
    cv2.destroyAllWindows()
    cap.release()
    print("face scan compleet .")


def analyse_user():
    print("analyzing ...")
    imgbase = face_recognition.load_image_file("moi.jpg")
    imgbase = cv2.cvtColor(imgbase, cv2.COLOR_BGR2GRAY)
    myface = face_recognition.face_locations(imgbase)[0]
    encodemyface = face_recognition.face_encodings(imgbase)[0]
    cv2.rectangle(imgbase,(myface[3],myface[0]),(myface[1],myface[2]),(255,0,255),2)


    sampleimg = face_recognition.load_image_file("photo.jpg")
    sampleimg = cv2.cvtColor(sampleimg, cv2.COLOR_BGR2GRAY)

    samplefacetest = face_recognition.face_locations(sampleimg)[0]

    try:
        encodesamplefacetest= face_recognition.face_encodings(sampleimg)[0]
    except IndexError as e:
        print("Index eurror . Authentification Failed")
        sys.exit()


    result = face_recognition.compare_faces([encodemyface],encodesamplefacetest)
    resulttest= str(result)

    if resulttest == "[True]":
        print("user Authentificated ..!")
    else:
        print("authentification failed .. !")


pren_photo()
analyse_user()








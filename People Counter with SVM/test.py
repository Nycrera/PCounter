from imutils.object_detection import non_max_suppression 
import numpy as np 
import imutils 
import cv2
import argparse

# Opencv pre-trained SVM with HOG people features 
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
LINE_1 = (0, 100)
LINE_2= (0, 200)

def detector(image):
    '''
    @image is a numpy array
    '''

    image = imutils.resize(image, width=min(400, image.shape[1]))
    # clone = image.copy()

    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8),
                                              padding=(32, 32), scale=1.05)

    # Applies non-max supression from imutils package to kick-off overlapped
    # boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return result

def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)
    if len(image) <= 0:
        print("[ERROR] could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)
    
    # shows the result
    for (xA, yA, xB, yB) in result:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (result, image)

def cameraDetect(cam):
    count_in = 0
    count_out = 0
    cap = cv2.VideoCapture(cam)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        cv2.line(frame, LINE_1, (frame.shape[1], LINE_1[1]), (255, 0, 0))
        cv2.line(frame, LINE_2, (frame.shape[1], LINE_2[1]), (255, 0, 0))
        result = detector(frame.copy())

        print((count_in, count_out))
        # shows the result
        
        for (xA, yA, xB, yB) in result:
            if (yA - LINE_1[1]) > (yA - LINE_2[1]):
                count_in+=1
            if (yA - LINE_1[1]) < (yA - LINE_2[1]):
                count_out+=1
            
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def detectPeople(args):
    image_path = args["image"]
    camera = args["camera"]

    # Routine to read local image
    if image_path != None and not camera:
        print("[INFO] Image path provided, attempting to read image")
        localDetect(image_path)

    # Routine to read images from webcam
    if camera:
        print("[INFO] reading camera images")
        cameraDetect(args["camera"])
    else:
        print("[INFO] No camera passed as argument.")
        print("Using default camera...")
        cameraDetect(0)

def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=None,
                    help="path to image test file directory")
    ap.add_argument("-c", "--camera", default=False,
                    help="Set as true if you wish to use the camera")
    args = vars(ap.parse_args())

    return args

def main():
    args = argsParser()
    detectPeople(args)


if __name__ == '__main__':
    main()

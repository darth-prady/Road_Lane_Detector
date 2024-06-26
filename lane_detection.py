import cv2
import numpy as np

# Defining the Region of Interest
def mask_region_img(img):
    l,b=img.shape
    polygon=np.array([[(150,l),(275,350),(400,325),(b,l)]])
    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,polygon,255)
    mask=cv2.bitwise_and(img,mask)
    return mask

# Averaging the detected lines through Hough Transform
def avg_lines(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        m = parameters[0]
        c = parameters[1]
        if m > 0:
            right.append((m, c))
        else:
            left.append((m, c))

    left_line = get_line(image, np.average(left, axis=0))
    right_line = get_line(image, np.average(right, axis=0))
    return np.array([left_line, right_line])

# Getting end-points of the lines
def get_line(image, average): 
    m, c = average 
    y1 = image.shape[0]
    y2 = int(y1 * 0.55)
    x1 = int((y1 - c) // m)
    x2 = int((y2 - c) // m)
    return np.array([x1, y1, x2, y2])

# Capturing Video file
video=cv2.VideoCapture("road.mp4")

while True:
    # Decoding video into frames (images) and downsizing the frames
    try:
        ret,frame=video.read()
        frame=cv2.resize(frame,(frame.shape[0],frame.shape[1]//2),)
        cv2.imshow("Video",frame)
    except:
        break

    # Converting image from RGB to Gray scale
    gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Noise reduction through Gaussian filter
    gaussian_blur = cv2.GaussianBlur(gray_scale,(5,5),cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Blur",gaussian_blur)

    # Canny Edge detection
    canny_edge = cv2.Canny(frame,100,200)
    cv2.imshow("Canny Edge Detection",canny_edge)

    # Finding the region of interest by using a mask
    mask=mask_region_img(canny_edge) 
    
    # Dilating the masked image
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.dilate(mask,kernel,iterations=1)
    cv2.imshow("Masked Image",mask)

    # Straight line detection through Hough Line Transform
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=2)
    lines=avg_lines(mask,lines)

    for line in lines:
        x1,y1,x2,y2 = line
        l=cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 10)
    
    result=cv2.bitwise_and(frame,l)
    cv2.imshow("Lane detection",result)

    if cv2.waitKey(1) & 0xFF==ord(' '):
        break

video.release()
cv2.destroyAllWindows
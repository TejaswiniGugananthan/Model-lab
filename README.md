```python
1.CONVERSIONS_OF-IMAGE:
i) Read and display the image:
 import cv2
 image=cv2.imread('flower.png',1)
 image=cv2.resize(image,(200,200))
 cv2.imshow('Tejaswini',image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
ii)  Write the image:
 import cv2
 image=cv2.imread('flower.png',0)
 cv2.imwrite('demos.png',image)
iii) Shape of the Image:
 import cv2
 image=cv2.imread('flower.png',1)
 print(image.shape)
iv)Access rows and columns:
 import random
 import cv2
 image=cv2.imread('flower.png',1)
 image=cv2.resize(image,(500,500))
 for i in range (250,500):
 for j in range(image.shape[1]):
        image[i][j]=[random.randint(0,255),
                     random.randint(0,255),
                     random.randint(0,255)] 
 cv2.imshow('part image',image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
v)Cut and paste portion of image:
 import cv2
 image=cv2.imread('flower.png',1)
 image=cv2.resize(image,(300,300))
 tag =image[150:200,110:160]
 image[110:160,150:200] = tag
 cv2.imshow('image1',image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
vi) BGR and RGB to HSV and GRAY
 import cv2
 img = cv2.imread('flower.png',1)
 img = cv2.resize(img,(200,200))
 cv2.imshow('Original Image',img)
 hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 cv2.imshow('BGR2HSV',hsv1)
 hsv2 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
 cv2.imshow('RGB2HSV',hsv2)
 gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 cv2.imshow('BGR2GRAY',gray1)
 gray2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
 cv2.imshow('RGB2GRAY',gray2)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
vii) HSV to RGB and BGR
 import cv2
 img = cv2.imread('flower.png')
 img = cv2.resize(img,(200,200))
 img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 cv2.imshow('Original HSV Image',img)
 RGB = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
 cv2.imshow('2HSV2BGR',RGB)
 BGR = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
 cv2.imshow('HSV2RGB',BGR)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
viii) RGB and BGR to YCrCb
 import cv2
 img = cv2.imread('flower.png')
 img = cv2.resize(img,(200,200))
 cv2.imshow('Original RGB Image',img)
 YCrCb1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
 cv2.imshow('RGB-2-YCrCb',YCrCb1)
 YCrCb2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
 cv2.imshow('BGR-2-YCrCb',YCrCb2)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
ix) Split and merge RGB Image
 import cv2
 img = cv2.imread('flower.png',1)
 img = cv2.resize(img,(200,200))
 R = img[:,:,2]
 G = img[:,:,1]
 B = img[:,:,0]
 cv2.imshow('R-Channel',R)
 cv2.imshow('G-Channel',G)
 cv2.imshow('B-Channel',B)
 merged = cv2.merge((B,G,R))
 cv2.imshow('Merged RGB image',merged)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
x) Split and merge HSV Image
 import cv2
 img = cv2.imread("flower.png",1)
 img = cv2.resize(img,(200,200))
 img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
 H,S,V=cv2.split(img)
 cv2.imshow('Hue',H)
 cv2.imshow('Saturation',S)
 cv2.imshow('Value',V)
 merged = cv2.merge((H,S,V))
 cv2.imshow('Merged',merged)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

2.Web camera:
i) Write the frame as JPG file :
 import cv2
 videoCaptureObject = cv2.VideoCapture(0)
 while (True):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("sample.jpeg",frame)
    result = False
 videoCaptureObject.release()
 cv2.destroyAllWindows()
ii) Display the video :
 import cv2
 videoCaptureObject = cv2.VideoCapture(0)
 while(True):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('myimage',frame)
    if cv2.waitKey(1) == ord('q'):
        break
 videoCaptureObject.release()
 cv2.destroyAllWindows()
iii) Display the video by resizing the window
 import cv2
 import numpy as np
 cap = cv2.VideoCapture(0)
 while True:
    ret, frame = cap.read() 
    width = int(cap.get(3))
    height = int(cap.get(4))
    image = np.zeros(frame.shape, np.uint8) 
    smaller_frame = cv2.resize(frame, (0,0), fx = 0.5, fy=0.5) 
    image[:height//2, :width//2] = smaller_frame
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = smaller_frame 
    image [height//2:, width//2:] = smaller_frame
    cv2.imshow('myimage', image)
    if cv2.waitKey(1) == ord('q'):
        break
 cap.release()
 cv2.destroyAllWindows()
iv) Rotate and display the video
 import cv2
 import numpy as np
 cap = cv2.VideoCapture(0)
 while True:
    ret, frame = cap.read() 
    width = int(cap.get(3))
    height = int(cap.get(4))
    image = np.zeros(frame.shape, np.uint8) 
    smaller_frame = cv2.resize(frame, (0,0), fx = 0.5, fy=0.5) 
    image[:height//2, :width//2] = cv2.rotate(smaller_frame,cv2.ROTATE_180)
    image[height//2:, :width//2] = cv2.rotate(smaller_frame,cv2.ROTATE_180)
    image[:height//2, width//2:] = smaller_frame 
    image [height//2:, width//2:] = smaller_frame
    cv2.imshow('myimage', image)
    if cv2.waitKey(1) == ord('q'):
        break
 cap.release()
 cv2.destroyAllWindows()


3. Histogram:
i) Input Grayscale Image and Color Image:
 import cv2
 import matplotlib.pyplot as plt
 import numpy as np
 gray_image = cv2.imread("flower.png")
 color_image = cv2.imread("flower 1.png",-1)
 cv2.imshow("Gray Image",gray_image)
 cv2.imshow("Colour Image",color_image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
ii)Histogram of Grayscale Image and any channel of Color Image:
 Gray_image = cv2.imread("flower.png")
 Color_image = cv2.imread("flower 1.png")
 gray_hist = cv2.calcHist([Gray_image],[0],None,[256],[0,256])
 color_hist = cv2.calcHist([Color_image],[0],None,[256],[0,256])
 plt.figure()
 plt.imshow(Gray_image)
 plt.show()
 plt.title("Histogram")
 plt.xlabel("Grayscale Value")
 plt.ylabel("Pixel Count")
 plt.stem(gray_hist)
 plt.show()
 plt.imshow(Color_image)
 plt.show()
 plt.title("Histogram of Color Image - Green Channel")
 plt.xlabel("Intensity Value")
 plt.ylabel("Pixel Count")
 plt.stem(color_hist)
 plt.show()
 cv2.waitKey(0)
iii)Histogram Equalization of Grayscale Image:
 gray_image = cv2.imread("flower.png",0)
 cv2.imshow('Grey Scale Image',gray_image)
 equ = cv2.equalizeHist(gray_image)
 cv2.imshow("Equalized Image",equ)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

4. IMAGE-TRANSFORMATIONS:
i)Image Translation:
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 input_image=cv2.imread("vijay.png")
 input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
 plt.axis('off')
 plt.imshow(input_image)
 plt.show()
 rows,cols,dim=input_image.shape
 M=np.float32([[1,0,50],  [0,1,100],  [0,0,1]])
 translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
 plt.axis('off')
 plt.imshow(translated_image)
 plt.show()
ii) Image Scaling:
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 org_image = cv2.imread("vijay.png")
 org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
 plt.imshow(org_image)
 plt.show()
 rows,cols,dim = org_image.shape
 M = np.float32([[1.5,0,0],[0,1.7,0],[0,0,1]])
 scaled_img = cv2.warpPerspective(org_image,M,(cols*2,rows*2))
 plt.imshow(org_image)
 plt.show()
iii)Image shearing:
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 org_image = cv2.imread("vijay.png")
 org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
 plt.imshow(org_image)
 plt.show()
 rows,cols,dim = org_image.shape
 M_X = np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
 M_Y = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
 sheared_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols*1.5),int(rows*1.5)))
 sheared_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols*1.5),int(rows*1.5)))
 plt.imshow(sheared_img_xaxis)
 plt.show()
 plt.imshow(sheared_img_yaxis)
 plt.show()
iv)Image Reflection
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 org_image = cv2.imread("vijay.png")
 org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
 plt.imshow(org_image)
 plt.show()
 rows,cols,dim = org_image.shape
 M_X = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
 M_Y = np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
 reflected_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols),int(rows)))
 reflected_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols),int(rows)))
 plt.imshow(reflected_img_xaxis)
 plt.show()
 plt.imshow(reflected_img_yaxis)
 plt.show()
v)Image Rotation
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 input_image = cv2.imread("vijay.png")
 input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
 plt.axis('off')
 plt.imshow(input_image)
 plt.show()
 angle=np.radians(10)
 M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,
 rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))
 plt.imshow(rotated_img)
 plt.show()
vi)Image Cropping
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 org_image = cv2.imread("vijay.png")
 org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
 plt.imshow(org_image)
 plt.show()
 rows,cols,dim = org_image.shape
 cropped_img=org_image[80:900,80:500]
 plt.imshow(cropped_img)
 plt.show()

5.Implementation-of-filter:
1. Smoothing Filters
 i) Using Averaging Filter
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 kernel = np.ones((11,11), np. float32)/121
 image3 = cv2.filter2D(image2, -1, kernel)
 plt.figure(figsize=(9,9))
 plt.subplot(1,2,1)
 plt.imshow(image2)
 plt.title('Orignal')
 plt.axis('off')
 plt.subplot(1,2,2)
 plt.imshow(image3)
 plt.title('Filtered')
 plt.axis('off')
 ii) Using Weighted Averaging Filter
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
 image4 = cv2.filter2D(image2, -1, kernel2)
 plt.imshow(image4)
 plt.title('Weighted Averaging Filtered')
iii) Using Gaussian Filter
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 gaussian_blur = cv2.GaussianBlur(src=image2, ksize=(11,11), sigmaX=0, sigmaY=0)
 plt.imshow(gaussian_blur)
 plt.title(' Gaussian Blurring Filtered')
 iv) Using Median Filter
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 median=cv2.medianBlur (src=image2, ksize=11)
 plt.imshow(median)
 plt.title(' Median Blurring Filtered')
 2. Sharpening Filters
 i) Using Laplacian Kernal
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
 image5 =cv2.filter2D(image2, -1, kernel3)
 plt.imshow(image5)
 plt.title('Laplacian Kernel')
 ii) Using Laplacian Operator
import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 image1 = cv2.imread('flower.png')
 image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
 new_image = cv2.Laplacian (image2, cv2.CV_64F)
 plt.imshow(new_image)
 plt.title('Laplacian Operator')

6.EDGE-DETECTION:
 import cv2
 import matplotlib.pyplot as plt
 img=cv2.imread("bouq.png",0)
 gray=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
 gray = cv2.GaussianBlur(gray,(3,3),0)

SOBEL EDGE DETECTOR
 i) SOBEL X AXIS
 sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
 plt.figure(figsize=(8,8))
 plt.subplot(1,2,1)
 plt.imshow(gray)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(sobelx)
 plt.title("Sobel X axis")
 plt.axis("off")
 plt.show()
 ii) SOBEL Y AXIS
 sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
 plt.figure(figsize=(8,8))
 plt.subplot(1,2,1)
 plt.imshow(gray)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(sobely)
 plt.title("Sobel Y axis")
 plt.axis("off")
 plt.show()
 iii) SOBEL XY AXIS
 sobelxy = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
 plt.figure(figsize=(8,8))
 plt.subplot(1,2,1)
 plt.imshow(gray)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(sobelxy)
 plt.title("Sobel XY axis")
 plt.axis("off")
 plt.show()
LAPLACIAN EDGE DETECTOR
 lap=cv2.Laplacian(gray,cv2.CV_64F)
 plt.figure(figsize=(8,8))
 plt.subplot(1,2,1)
 plt.imshow(gray)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(lap)
 plt.title("Laplacian Edge Detector")
 plt.axis("off")
 plt.show()
 CANNY EDGE DETECTOR
 canny=cv2.Canny(gray,120,150)
 plt.figure(figsize=(8,8))
 plt.subplot(1,2,1)
 plt.imshow(gray)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(canny)
 plt.title("Canny Edge Detector")
 plt.axis("off")
 plt.show()

7.Edge-Linking-using-Hough-Transformm:
i)Input image and grayscale image
 import numpy as np
 import cv2
 import matplotlib.pyplot as plt
 img=cv2.imread("duke.png",0)
 img_c=cv2.imread("duke.png",1)
 img_c=cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB)
 gray=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
 gray = cv2.GaussianBlur(gray,(3,3),0)
 plt.figure(figsize=(13,13))
 plt.subplot(1,2,1)
 plt.imshow(img_c)
 plt.title("Original Image")
 plt.axis("off")
 plt.subplot(1,2,2)
 plt.imshow(gray)
 plt.title("Gray Image")
 plt.axis("off")
 plt.show()
ii) Canny Edge detector 
 canny=cv2.Canny(gray,120,150)
 plt.imshow(canny)
 plt.title("Canny Edge Detector")
 plt.axis("off")
 plt.show()
iii) Detect points that form a line using HoughLinesP
 lines=cv2.HoughLinesP(canny,1,np.pi/180,threshold=80,minLineLength=50,maxLineGap=250)
iv) Draw lines on the image
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img_c,(x1,y1),(x2,y2),(255,0,0),3)
Display the result of Hough transform
plt.imshow(img_c)
plt.title("Result Image")
plt.axis("off")
plt.show()

8.THRESHOLDING
i) Load the necessary packages:
import numpy as np
import matplotlib.pyplot as plt
import cv2

ii) Read the Image and convert to grayscale
image = cv2.imread("nature.png",1)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_gray = cv2.imread("nature.png",0)

iii) Use Global thresholding to segment the image
ret,thresh_img1=cv2.threshold(image_gray,86,255,cv2.THRESH_BINARY)
ret,thresh_img2=cv2.threshold(image_gray,86,255,cv2.THRESH_BINARY_INV)
ret,thresh_img3=cv2.threshold(image_gray,86,255,cv2.THRESH_TOZERO)
ret,thresh_img4=cv2.threshold(image_gray,86,255,cv2.THRESH_TOZERO_INV)
ret,thresh_img5=cv2.threshold(image_gray,100,255,cv2.THRESH_TRUNC)

iv) Use Adaptive thresholding to segment the image
thresh_img7=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
thresh_img8=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
v) Use Otsu's method to segment the image
ret,thresh_img6=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 vi) Display the results
titles=["Gray Image","Threshold Image (Binary)","Threshold Image (Binary Inverse)","Threshold Image (To Zero)"
       ,"Threshold Image (To Zero-Inverse)","Threshold Image (Truncate)","Otsu","Adaptive Threshold (Mean)","Adaptive Threshold (Gaussian)"]
images=[image_gray,thresh_img1,thresh_img2,thresh_img3,thresh_img4,thresh_img5,thresh_img6,thresh_img7,thresh_img8]
for i in range(0,9):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title(titles[i])
    plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

9. EROSION AND DILATION:
import numpy as np
import cv2
import matplotlib.pyplot as plt
i) Create the Text using cv2.putText
img = np.zeros((100,400),dtype = 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img ,'CUTE',(60,70),font,2,(255),5,cv2.LINE_AA)
plt.imshow(img)
plt.axis('off')

ii) Create the structuring element
kernel = np.ones((5,5),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
cv2.erode(img,kernel)

iii) Erode the image
img_erode = cv2.erode(img,kernel1)
plt.imshow(img_erode)
plt.axis('off')

iv) Dilate the image
img_dilate = cv2.dilate(img,kernel1)
plt.imshow(img_dilate)
plt.axis('off')

10. OPENING AND CLOSING:

import cv2
import numpy as np
i) Create the Text using cv2.putText
img = np.zeros((350, 1400), dtype='uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Telecast', (15, 200), font, 5, (255), 10, cv2.LINE_AA)
cv2.imshow('created_text', img)
cv2.waitKey(0)

ii) Create the structuring element
struct_ele = np.ones((9, 9), np.uint8)
Use Opening operation
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, struct_ele)
cv2.imshow('Opening', opening)
cv2.waitKey(0)

iii) Use Closing Operation
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct_ele)
cv2.imshow('Closing', closing)
cv2.waitKey(0)

11. HUFFMAN CODING:

i) Get the input String
string = 'TEJASWINI G'
ii) Create tree nodes
class NodeTree(object):
    
    def __init__(self, left=None, right=None):
        self.left = left 
        self.right  = right

    def children(self):
        return (self.left,self.right)
iii) Main function to implement huffman coding
def huffman_code_tree(node, left=True, binString=''): 
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    D= dict()
    D.update(huffman_code_tree(l, True, binString + '0'))
    D.update(huffman_code_tree(r, False, binString + '1'))
    return D
iv) Calculate frequency of occurrence
freq = {}

for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    
    nodes = sorted(nodes, key=lambda x: x[1],reverse=True)

v) Print the characters and its huffmancode
huffmanCode = huffman_code_tree (nodes[0][0])
print('Char | Huffman code ')
print('---------------------')

for (char, frequency) in freq:
    print('%-4r %12s' % (char, huffmanCode[char]))
```


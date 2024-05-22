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



```


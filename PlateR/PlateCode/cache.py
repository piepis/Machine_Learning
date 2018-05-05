import cv2
import hashlib
def verticalMappingToFolder(image):
    name = hashlib.md5(image.data).hexdigest()[:8]
    cv2.imwrite("./cache/finemapping/"+name+".png",image)



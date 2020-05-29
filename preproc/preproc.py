import cv2
import numpy as np 

from pathlib import Path 
from typing import Callable, Any

Image = [np.uint8]

data_dir = './data/image-data/'
img_format = 'jpg'


# resize image to desired dimension maintaining it's original ratio
def resize(img: Image, low_dim: int=1024) -> Image:
    dim_ratio = img.shape[0] / img.shape[1]
    return cv2.resize(img, (low_dim, int(dim_ratio * low_dim)))


# display image until user presses q from keyboard
def display(img: Image, title: str='display') -> None:
    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()



# temp
def play(img: Image) -> Image:
    resized = resize(img)
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    sobel = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=5)
    bitwise = cv2.bitwise_and(laplacian, sobel)
    display(bitwise)


# do entire dir
def process_imgs_from_dir(data_dir: str, img_format: str, process: Callable[Image, Any]) -> None:
    pathlist = Path(data_dir).glob('**/*-binarized.' + img_format)
    for path in pathlist:
        img = cv2.imread(str(path)) # read from source
        process(img)


#process_imgs_from_dir(data_dir=data_dir, img_format=img_format, process=play)
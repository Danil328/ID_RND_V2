from utils.face_detection.face_detection import get_face
import glob
import os
from tqdm import trange, tqdm
import cv2

PATH_TO_IMAGE = '../data/val'
PATH_TO_SAVE_IMAGE = '../data/val_crop'

if __name__ == '__main__':
	images = glob.glob(os.path.join(PATH_TO_IMAGE, '*/*/*.png'))
	for image in tqdm(images):
		path_to_new_image = image.replace(PATH_TO_IMAGE, PATH_TO_SAVE_IMAGE)
		if os.path.exists(path_to_new_image):
			continue
		im = cv2.imread(image)
		crop_im = get_face(im)
		if not os.path.isdir(os.path.dirname(path_to_new_image)):
			os.mkdir(os.path.dirname(path_to_new_image))
		cv2.imwrite(path_to_new_image, crop_im)



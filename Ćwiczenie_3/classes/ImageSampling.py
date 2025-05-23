import os
import cv2

class ImageSampling:

    def __init__(self, input_dir, output_dir, image_size, crop_size):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.crop_size = crop_size

    def load_and_resize_images(self):
        for subdir, dirs, files in os.walk(self.input_dir):
            for file in files:
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.image_size)
                self.crop_and_save(img, file)

    def crop_and_save(self, image, category):
        category = os.path.splitext(category)[0]

        num_crops = int(self.image_size[0] / self.crop_size[0])
        category_dir = os.path.join(self.output_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        for i in range(num_crops):
            for j in range(num_crops):
                crop_img = image[i * self.crop_size[0]:(i + 1) * self.crop_size[0],
                           j * self.crop_size[1]:(j + 1) * self.crop_size[1]]
                crop_path = os.path.join(category_dir, f'crop_{i}_{j}.jpg')
                cv2.imwrite(crop_path, crop_img)

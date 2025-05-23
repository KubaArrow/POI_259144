import os
import numpy as np
from skimage import io, color
from skimage.feature import *
import pandas as pd
class TextureExtractor:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def display_directories(self):
        directories = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
        for i, directory in enumerate(directories):
            print(f"{i + 1}. {directory}")
        print(f"{len(directories) + 1}. All directories")
        return directories

    def select_directory(self, directories):
        directory_index = int(input("Select directory by entering the corresponding number: ")) - 1
        if directory_index == len(directories):
            return None
        selected_directory = directories[directory_index]
        print(f"Selected directory: {selected_directory}")
        return selected_directory

    def load_images_from_directory(self, directory):
        dir_path = os.path.join(self.input_dir, directory)
        images = [io.imread(os.path.join(dir_path, img)) for img in os.listdir(dir_path) if img.endswith(".jpg")]
        return images

    def convert_to_gray(self, image):
        gray_image = color.rgb2gray(image)
        gray_image = np.round(gray_image * 63).astype(np.uint8)
        return gray_image

    def calculate_features(self, images, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        features = []
        for image in images:
            gray_image = self.convert_to_gray(image)
            feature_vector = []
            for distance in distances:
                for angle in angles:
                    glcm = graycomatrix(gray_image, distances=[distance], angles=[angle], levels=64, symmetric=True,
                                        normed=True)
                    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
                    correlation = graycoprops(glcm, 'correlation').mean()
                    contrast = graycoprops(glcm, 'contrast').mean()
                    energy = graycoprops(glcm, 'energy').mean()
                    homogeneity = graycoprops(glcm, 'homogeneity').mean()
                    asm = graycoprops(glcm, 'ASM').mean()
                    feature_vector.extend([dissimilarity, correlation, contrast, energy, homogeneity, asm])
            features.append(feature_vector)
        return features

    def extract_features(self, selected_directory=None):
        if selected_directory:
            directories = [selected_directory]
        else:
            directories = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]

        for directory in directories:
            images = self.load_images_from_directory(directory)
            all_categories = [directory] * len(images)
            features = self.calculate_features(images)

            feature_names = []
            distances = [1, 3, 5]
            angles = [0, round(np.pi / 4, 2), round(np.pi / 2, 2), round(3 * np.pi / 4, 2)]
            for distance in distances:
                for angle in angles:
                    for feature in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'asm']:
                        feature_name = f"{feature}_d{distance}_a{angle}"
                        feature_names.append(feature_name)

            df = pd.DataFrame(features, columns=feature_names)
            df['category'] = all_categories

            output_dir = "./output_csv"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{directory}_textures.csv")
            df.to_csv(output_path, index=False)
            print(f"Texture features saved to: {output_path}")

    def extract_features_merged(self):
        directories = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
        all_features = []
        all_labels = []

        for directory in directories:
            images = self.load_images_from_directory(directory)
            features = self.calculate_features(images)
            all_features.extend(features)
            all_labels.extend([directory] * len(features))

        feature_names = []
        distances = [1, 3, 5]
        angles = [0, round(np.pi / 4, 2), round(np.pi / 2, 2), round(3 * np.pi / 4, 2)]
        for distance in distances:
            for angle in angles:
                for feature in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'asm']:
                    feature_name = f"{feature}_d{distance}_a{angle}"
                    feature_names.append(feature_name)

        df = pd.DataFrame(all_features, columns=feature_names)
        df['category'] = all_labels

        output_dir = "./output_csv"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "all_textures.csv")
        df.to_csv(output_path, index=False)
        print(f"Combined texture features saved to: {output_path}")


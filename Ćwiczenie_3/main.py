import argparse
from classes.ImageSampling import ImageSampling
from classes.TextureExtraction import TextureExtractor
from classes.Classifier import Classifier
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Texture classification pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command")

    # Command 1: Create texture samples
    parser_sample = subparsers.add_parser("sample", help="Load textures and make texture samples")
    parser_sample.add_argument("--input", type=str, default="img", help="Input directory with images")
    parser_sample.add_argument("--output", type=str, default="samples", help="Output directory for samples")
    parser_sample.add_argument("--image_width", type=int, default=512)
    parser_sample.add_argument("--image_height", type=int, default=512)
    parser_sample.add_argument("--crop_width", type=int, required=True)
    parser_sample.add_argument("--crop_height", type=int, required=True)

    # Command 2: Extract texture features
    parser_extract = subparsers.add_parser("extract", help="Create and save vectors to .csv")
    parser_extract.add_argument("--input", type=str, default="samples", help="Directory with texture samples")
    parser_extract.add_argument("--directory", type=str, help="Specific subdirectory to process (optional)")
    parser_extract.add_argument("--merge", action="store_true", help="Merge all directories into one CSV file")

    # Command 3: Classify feature vectors
    parser_classify = subparsers.add_parser("classify", help="Classify feature vectors using SVM")
    parser_classify.add_argument("--csv", type=str, help="Path to specific CSV file in output_csv/")

    args = parser.parse_args()

    if args.command == "sample":
        sampler = ImageSampling(
            input_dir=args.input,
            output_dir=args.output,
            image_size=(args.image_width, args.image_height),
            crop_size=(args.crop_width, args.crop_height)
        )
        sampler.load_and_resize_images()

    elif args.command == "extract":
        extractor = TextureExtractor(args.input)
        if args.directory:
            extractor.extract_features(selected_directory=args.directory)
        elif args.merge:
            extractor.extract_features_merged()
        else:
            extractor.extract_features()

    elif args.command == "classify":

        def classify_from_dataframe(df, filename):
            feature_names = df.columns[:-1]
            label_name = df.columns[-1]
            features = df[feature_names].values
            labels = df[label_name].values
            if len(set(labels)) < 2:
                print(f"Skipping '{filename}' — only one class present.")
                return
            print(f"Classifying '{filename}'...")
            classifier = Classifier(features, labels)
            classifier.classify()
        output_dir = "output_csv"

        if args.csv:
            csv_path = os.path.join(output_dir, args.csv)
            if not os.path.exists(csv_path):
                print(f"File '{args.csv}' not found in '{output_dir}'.")
            else:
                df = pd.read_csv(csv_path)
                classify_from_dataframe(df, args.csv)
        else:
            for file in os.listdir(output_dir):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(output_dir, file))
                    classify_from_dataframe(df, file)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

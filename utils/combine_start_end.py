from tqdm import tqdm
import cv2
import argparse
from pathlib import Path
import numpy as np

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="load an decode the json_predictions to arrays")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-b", "--baseline", help="Baseline", required=True, type=str)
    io_args.add_argument("-s", "--start", help="Start", required=True, type=str)
    io_args.add_argument("-e", "--end", help="End", required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output file", required=True, type=str)
    
    args = parser.parse_args()
    return args

def main(args):
    baseline_path = Path(args.baseline)
    start_path = Path(args.start)
    end_path = Path(args.end)
    
    output_path = Path(args.output)
    
    if not baseline_path.is_dir():
        raise FileNotFoundError(f"Baseline folder ({baseline_path}) not found")
    if not start_path.is_dir():
        raise FileNotFoundError(f"Start folder ({start_path}) not found")
    if not end_path.is_dir():
        raise FileNotFoundError(f"End folder ({end_path}) not found")
    
    if not output_path.is_dir():
        print(
            f"Could not find output dir ({output_path}), creating one at specified location")
        output_path.mkdir(parents=True)
    
    baseline_image_paths = list(baseline_path.glob("*.png"))
    if len(baseline_image_paths) == 0:
        raise FileNotFoundError(f"No png images found in folder {baseline_path}")
    
    start_image_paths = [start_image_path for path in baseline_image_paths if (start_image_path := start_path.joinpath(path.name)).is_file()]
    end_image_paths = [end_image_path for path in baseline_image_paths if (end_image_path := end_path.joinpath(path.name)).is_file()]
    
    if len(start_image_paths) != len(baseline_image_paths):
        raise FileNotFoundError(f"Number of images in {start_path} does not match number of images in {baseline_path}")
    if len(end_image_paths) != len(baseline_image_paths):
        raise FileNotFoundError(f"Number of images in {start_path} does not match number of images in {baseline_path}")
    
    print("Combining Images")
    # TODO Multithread
    for i in tqdm(range(len(baseline_image_paths))):
        baseline_image_path = baseline_image_paths[i]
        start_image_path = start_image_paths[i]
        end_image_path = end_image_paths[i]
        
        baseline_image = cv2.imread(str(baseline_image_path), cv2.IMREAD_GRAYSCALE)
        start_image = cv2.imread(str(start_image_path), cv2.IMREAD_GRAYSCALE)
        end_image = cv2.imread(str(end_image_path), cv2.IMREAD_GRAYSCALE)
        
        image = np.stack([end_image, start_image, baseline_image], axis=-1)
        # image = image[..., ::-1] #Flip for BGR
        
        output_image_path = output_path.joinpath(baseline_image_path.name)
        # print(output_image_path)
        
        cv2.imwrite(str(output_image_path), image)
        
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)
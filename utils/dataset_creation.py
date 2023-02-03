import argparse
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from natsort import os_sorted
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode
from utils.path_utils import image_path_to_xml_path, xml_path_to_image_path

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copying files from multiple folders into a single structured dataset")
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="input folders",
                            nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder",
                        required=True, type=str)
    
    parser.add_argument("-c", "--copy", action="store_true", )
    parser.add_argument("-m", "--mode", choices=["link", "symlink", "copy"], help="Mode for moving the images", default='copy')
    
    args = parser.parse_args()
    return args


def copy_xml_paths(xml_paths: list[Path], output_dir: Path, mode="copy") -> list[Path]:
    """
    copy a list of pageXML paths to an output dir. The respective images are also copied

    Args:
        xml_paths (list[Path]): image paths
        output_dir (Path): path of the output dir
        mode (str, optional): type of copy mode (symlink, link, copy). Defaults to "copy".

    Returns:
        list[Path]: output paths
    """
    if not output_dir.is_dir():
        print(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)
    
    page_dir = output_dir.joinpath("page")
    
    if not page_dir.is_dir():
        print(f"Could not find output page dir ({page_dir}), creating one at specified location")
        page_dir.mkdir(parents=True)
        
    output_paths = []
    for xml_path in xml_paths:
        image_path = xml_path_to_image_path(xml_path)
        output_image_path = output_dir.joinpath(image_path.name)
        output_xml_path = page_dir.joinpath(xml_path.name)
        copy_mode(image_path, output_image_path, mode=mode)
        copy_mode(xml_path, output_xml_path, mode=mode)
        
        output_paths.append(output_xml_path)
        
    return output_paths

def copy_image_paths(image_paths: list[Path], output_dir: Path, mode="copy") -> list[Path]:
    """
    copy a list of image paths to an output dir. The respective pageXMLs are also copied

    Args:
        image_paths (list[Path]): image paths
        output_dir (Path): path of the output dir
        mode (str, optional): type of copy mode (symlink, link, copy). Defaults to "copy".

    Returns:
        list[Path]: output paths
    """
    if not output_dir.is_dir():
        print(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)
    
    page_dir = output_dir.joinpath("page")
    
    if not page_dir.is_dir():
        print(f"Could not find output page dir ({page_dir}), creating one at specified location")
        page_dir.mkdir(parents=True)
        
    output_paths = []
    for image_path in image_paths:
        xml_path = image_path_to_xml_path(image_path)
        output_image_path = output_dir.joinpath(image_path.name)
        output_xml_path = page_dir.joinpath(xml_path.name)
        copy_mode(image_path, output_image_path, mode=mode)
        copy_mode(xml_path, output_xml_path, mode=mode)
        
        output_paths.append(output_image_path)
        
    return output_paths

def main(args):
    """
    Create a dataset structure (train, val, test) from multiple sub dirs

    Args:
        args (argparse.Namespace): arguments for where to find the images, and the output dir

    Raises:
        ValueError: must give an input
        ValueError: must give an output
        FileNotFoundError: input dir is missing
        ValueError: found duplicates in the images
        FileNotFoundError: no images found in sub dir
    """
    if args.input == []:
        raise ValueError("Must give an input")
    if args.output == "":
        raise ValueError("Must give an output")
    
    input_dirs = [Path(path) for path in args.input]
    
    
    # image_formats = [".bmp", ".dib",
    #                  ".jpeg", ".jpg", ".jpe",
    #                  ".jp2",
    #                  ".png",
    #                  ".webp",
    #                  ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
    #                  ".pfm",
    #                  ".sr", ".ras",
    #                  ".tiff", ".tif",
    #                  ".exr",
    #                  ".hdr", ".pic"]
    
    
    # image_formats = ['.jpg']
    
    all_xml_paths = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"{input_dir} does not exist")
        # Find all images
        
        xml_paths = list(input_dir.rglob(f"**/*.xml"))
        if len(xml_paths) == 0:
            raise FileNotFoundError(f"No xml_files found within {input_dir}")
        all_xml_paths.extend(xml_paths) # Assume depth 2
        
    all_image_paths = [xml_path_to_image_path(path).absolute() for path in all_xml_paths]
    
    
    if len(all_image_paths) != len(set(path.stem for path in all_image_paths)):
        duplicates = {k:v for k, v in Counter(path.stem for path in all_image_paths).items() if v > 1}
        
        print(os_sorted(duplicates.items(), key=lambda s: s[0]))
        raise ValueError("Found duplicate stems for images")
    
    
    train_paths, val_test_paths = train_test_split(all_image_paths, test_size=0.2)
    
    val_paths, test_paths = train_test_split(val_test_paths, test_size=0.5)
    
    print("Number of train images:", len(train_paths))
    print("Number of validation images:", len(val_paths))
    print("Number of test images:", len(test_paths))
    
    output_dir = Path(args.output)

    if not output_dir.is_dir():
        print(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)
    
    train_dir = output_dir.joinpath("train")
    val_dir = output_dir.joinpath("val")
    test_dir = output_dir.joinpath("test")
    
    if args.copy:
        train_paths = copy_image_paths(train_paths, train_dir, mode=args.mode)
        val_paths = copy_image_paths(val_paths, val_dir, mode=args.mode)
        test_paths = copy_image_paths(test_paths, test_dir, mode=args.mode)
        
        train_paths = [path.relative_to(output_dir) for path in train_paths]
        val_paths = [path.relative_to(output_dir) for path in val_paths]
        test_paths= [path.relative_to(output_dir) for path in test_paths]
    
    
    with open(output_dir.joinpath("train_filelist.txt"), mode='w') as f:
        for train_path in train_paths:
            f.write(f"{train_path}\n")
    with open(output_dir.joinpath("val_filelist.txt"), mode='w') as f:
        for val_path in val_paths:
            f.write(f"{val_path}\n")
    with open(output_dir.joinpath("test_filelist.txt"), mode='w') as f:
        for test_path in test_paths:
            f.write(f"{test_path}\n")
            
    with open(output_dir.joinpath("info.txt"), mode='w') as f:
        f.write(f"Created: {datetime.now()}\n")
        f.write(f"Number of train images: {len(train_paths)}\n")
        f.write(f"Number of validation images: {len(val_paths)}\n")
        f.write(f"Number of test images: {len(test_paths)}\n")
        
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)
import argparse
from input_utils import get_file_paths
from utils.path_utils import image_path_to_xml_path
    
def get_arguments() -> argparse.Namespace:    
    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file",
                            nargs="+", action="extend", type=str)
    io_args.add_argument("-o", "--output", help="Output folder",
                        required=True, type=str)
    args = parser.parse_args()
    return args

def main(args):
    
    # Formats found here: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#imread
    image_formats = [".bmp", ".dib",
                     ".jpeg", ".jpg", ".jpe",
                     ".jp2",
                     ".png",
                     ".webp",
                     ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
                     ".pfm",
                     ".sr", ".ras",
                     ".tiff", ".tif",
                     ".exr",
                     ".hdr", ".pic"]

    image_paths = get_file_paths(args.input, image_formats)
    xml_paths = [image_path_to_xml_path(image_path) for image_path in image_paths]

if __name__ == "__main__":
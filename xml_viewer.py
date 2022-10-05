import argparse

from detectron2.data import MetadataCatalog, Metadata
from detectron2.utils.visualizer import Visualizer

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    parser.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)
    
    args = parser.parse_args()
    return args

def main(args) -> None:
    pass

if __name__ == "__main__":
    args = get_arguments()
    main(args)
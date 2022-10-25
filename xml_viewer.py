from multiprocessing import Pool
from tqdm import tqdm
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

from natsort import os_sorted
from detectron2.data import MetadataCatalog, Metadata
from detectron2.utils.visualizer import Visualizer

from page_xml.xml_to_image import XMLImage
from utils.path import clean_input, xml_path_to_image_path

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLImage.get_parser()],
        description="Preprocessing an annotated dataset of documents with pageXML")
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/files", nargs="+", default=[],
                        required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)
    
    parser.add_argument("-v", "--variation", help="Output mode",
                        choices=["gray", "color", "overlay"], default="overlay", type=str)
    
    args = parser.parse_args()
    return args

class Viewer:
    def __init__(self, xml_to_image: XMLImage, output_dir: str|Path, mode='gray') -> None:
        
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        self.output_dir: Path = output_dir
        self.xml_to_image: XMLImage = xml_to_image
        self.metadata = Metadata()
        
        region_names = xml_to_image.get_regions()
        region_colors = [(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)]
        
        assert len(region_names) == len(region_colors), "Colors must match names in length"
        
        self.metadata.set(stuff_classes=region_names,
                          stuff_colors=region_colors,
                          evaluator_type="sem_seg",
                          ignore_label=255)
        
        self.mode = mode
        
        if self.mode == 'gray':
            self.save_function = self.save_gray_image
        elif self.mode == 'color':
            self.save_function = self.save_color_image
        elif self.mode == 'overlay':
            self.save_function = self.save_overlay_image
        else:
            raise NotImplementedError
        
    def save_gray_image(self, xml_path_i: Path):
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".png")
        gray_image = self.xml_to_image.run(xml_path_i)
        cv2.imwrite(str(output_image_path), gray_image)
        
    def save_color_image(self, xml_path_i: Path):
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".png")
        gray_image = self.xml_to_image.run(xml_path_i)
        
        color_image = np.empty((*gray_image.shape, 3), dtype=np.uint8)
        
        colors = self.metadata.get("stuff_colors")
        assert colors is not None, "Can't make color images without colors"
        assert np.max(gray_image) < len(colors), "Not enough colors, grayscale has too many classes"
        
        for i, color in enumerate(colors):
            color_image[gray_image == i] = np.asarray(color).reshape((1,1,3))
            
        cv2.imwrite(str(output_image_path), color_image[..., ::-1])
    
    @staticmethod
    def check_image_exists(xml_paths: list[Path]):
        _ = [xml_path_to_image_path(xml_path) for xml_path in xml_paths]
                
    def save_overlay_image(self, xml_path_i: Path):
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".png")
        gray_image = self.xml_to_image.run(xml_path_i)
        
        image_path_i = xml_path_to_image_path(xml_path_i)
        
        image = cv2.imread(str(image_path_i))
        
        vis_im = Visualizer(image[:, :, ::-1].copy(),
                            metadata=self.metadata,
                            scale=1
                            )
        vis_im = vis_im.draw_sem_seg(gray_image)
        overlay_image = vis_im.get_image()
        cv2.imwrite(str(output_image_path), overlay_image[..., ::-1])
        
    def run(self, xml_list: list[Path] | list[str]):
        cleaned_xml_list: list[Path] = [Path(path) if isinstance(path, str) else path for path in xml_list]
        cleaned_xml_list = [path.resolve() for path in cleaned_xml_list]
        
        if self.mode == 'overlay':
            self.check_image_exists(cleaned_xml_list)
            
        if not self.output_dir.is_dir():
            print(f"Could not find output dir ({self.output_dir}), creating one at specified location")
            self.output_dir.mkdir(parents=True)
        
        # Single thread
        # for xml_path_i in tqdm(cleaned_xml_list):
        #     self.save_function(xml_path_i)
        
        # Multithread
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(
                self.save_function, cleaned_xml_list), total=len(cleaned_xml_list)))
        
def main(args) -> None:
    xml_list = clean_input(args.input, suffixes=[".xml"])
    
    xml_to_image = XMLImage(
        mode=args.mode,
        line_width=args.line_width,
        line_color=args.line_color,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type
    )
    
    viewer = Viewer(xml_to_image=xml_to_image, output_dir=args.output, mode=args.variation)
    viewer.run(xml_list)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
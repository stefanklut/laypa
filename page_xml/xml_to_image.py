import argparse
from pathlib import Path
import numpy as np

from .xmlPAGE import PageData
from .xml_regions import XMLRegions
from typing import Optional


def get_arguments() -> argparse.Namespace:
    # HACK hardcoded regions if none are given
    republic_regions = ["marginalia", "page-number", "resolution", "date",
                        "index", "attendance", "Resumption", "resumption", "Insertion", "insertion"]
    republic_merge_regions = [
        "resolution:Resumption,resumption,Insertion,insertion"]

    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    parser.add_argument("-i", "--input", help="Input file",
                        required=True, type=str)
    parser.add_argument(
        "-o", "--output", help="Output file", required=True, type=str)
    parser.add_argument("-m", "--mode", help="Output mode",
                        choices=["baseline", "region", "both"], default="baseline", type=str)

    parser.add_argument("-w", "--line_width",
                        help="Used line width", type=int, default=5)
    parser.add_argument("-c", "--line_color", help="Used line color",
                        choices=list(range(256)), type=int, metavar="{0-255}", default=1)

    parser.add_argument(
        "--regions",
        default=republic_regions,
        nargs="+",
        type=str,
        help="""List of regions to be extracted. 
                            Format: --regions r1 r2 r3 ...""",
    )
    parser.add_argument(
        "--merge_regions",
        default=republic_merge_regions,
        nargs="+",
        type=str,
        help="""Merge regions on PAGE file into a single one.
                            Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                            will be merged into r1 and r5 into r4""",
    )
    parser.add_argument(
        "--region_type",
        default=None,
        nargs="+",
        type=str,
        help="""Type of region on PAGE file.
                            Format --region_type t1:r1,r3 t2:r5, then type t1
                            will assigned to regions r1 and r3 and type t2 to
                            r5 and so on...""",
    )

    args = parser.parse_args()
    return args


class XMLImage(XMLRegions):
    def __init__(self, mode: str, 
                 line_width: Optional[int] = None, 
                 line_color: Optional[int] = None, 
                 regions: Optional[list[str]] = None, 
                 merge_regions: Optional[list[str]] = None, 
                 region_type: Optional[list[str]] = None) -> None:
        super().__init__(mode, line_width, line_color, regions, merge_regions, region_type)

    def run(self, xml_path: Path, image_shape=None) -> np.ndarray:
        gt_data = PageData(xml_path)
        gt_data.parse()

        if image_shape is None:
            image_shape = gt_data.get_size()[::-1]

        if self.mode == "baseline":
            baseline_mask = gt_data.build_baseline_mask(image_shape,
                                                        color=self.line_color,
                                                        line_width=self.line_width)
            mask = baseline_mask
        elif self.mode == "region":
            region_mask = gt_data.build_mask(image_shape,
                                             set(self.region_types.values()),
                                             self.region_classes)
            mask = region_mask
        else:
            raise NotImplementedError

        return mask



if __name__ == "__main__":
    args = get_arguments()
    XMLImage(
        mode=args.mode,
        line_width=args.line_width,
        line_color=args.line_color,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    
    
    

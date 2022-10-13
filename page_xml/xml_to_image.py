import argparse
from pathlib import Path

from .xmlPAGE import PageData
from typing import Optional


def get_arguments() -> argparse.Namespace:
    # HACK hardcoded regions if none are given
    republic_regions = ["marginalia", "page-number", "resolution", "date",
                        "index", "attendance", "Resumption", "resumption", "Insertion", "insertion"]
    republic_merge_regions = [
        "resolution:Resumption,resumption,Insertion,insertion"]

    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    parser.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)
    parser.add_argument(
        "-o", "--output", help="Output folder", required=True, type=str)
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


class XMLImage:
    def __init__(self,
                 mode,
                 line_width=None,
                 line_color=None,
                 regions=None,
                 merge_regions=None,
                 region_type=None) -> None:
        self.mode = mode
        if self.mode == "baseline":
            assert line_width is not None
            assert line_color is not None

            self.line_width = line_width
            self.line_color = line_color

        elif self.mode == "region":
            assert regions is not None

            # regions: list of type names (required for lookup)
            # merge_regions: regions to be merged. r1:r2,r3  -> r2 and r3 become region r1
            # region_type: type per_region. t1:r1,r2  -> r1 and r2 become type t1
            self._regions: list[str] = regions
            self._merge_regions: Optional[list[str]] = merge_regions
            self._region_type: Optional[list] = region_type

            self.region_classes = self._build_class_regions()
            self.region_types = self._build_region_types()
            self.merged_regions = self._build_merged_regions()
            if self.merged_regions is not None:
                for parent, childs in self.merged_regions.items():
                    for child in childs:
                        self.region_classes[child] = self.region_classes[parent]
        else:
            raise NotImplementedError

    def _build_class_regions(self) -> dict:
        """given a list of regions assign a equaly separated class to each one"""

        class_dic = {}

        for c, r in enumerate(self._regions):
            class_dic[r] = c + 1
        return class_dic

    def _build_merged_regions(self) -> Optional[dict]:
        """build dic of regions to be merged into a single class"""
        if self._merge_regions is None:
            return None
        to_merge = {}
        msg = ""
        for c in self._merge_regions:
            try:
                parent, childs = c.split(":")
                if parent in self._regions:
                    to_merge[parent] = childs.split(",")
                else:
                    msg = '\nRegion "{}" to merge is not defined as region'.format(
                        parent
                    )
                    raise
            except:
                raise argparse.ArgumentTypeError(
                    "Malformed argument {}".format(c) + msg
                )

        return to_merge

    def _build_region_types(self) -> dict:
        """ build a dic of regions and their respective type"""
        reg_type = {"full_page": "TextRegion"}
        if self._region_type is None:
            for reg in self._regions:
                reg_type[reg] = "TextRegion"
            return reg_type
        msg = ""
        for c in self._region_type:
            try:
                parent, childs = c.split(":")
                regs = childs.split(",")
                for reg in regs:
                    if reg in self._regions:
                        reg_type[reg] = parent
                    else:
                        msg = '\nCannot assign region "{0}" to any type. {0} not defined as region'.format(
                            reg
                        )
            except:
                raise argparse.ArgumentTypeError(
                    "Malformed argument {}".format(c) + msg
                )
        return reg_type
    
    def get_regions(self) -> list:
        remaining_regions = ["background"]
        if self.mode == 'region':
            if self.merged_regions is None:
                raise ValueError("merged_regions is not set")
            removed_regions = set()
            for values in self.merged_regions.values():
                removed_regions = removed_regions.union(set(values))
            remaining_regions.extend(region for region in self._regions if not region in removed_regions)
        else:
            remaining_regions.extend(["baseline"])
        
        return remaining_regions
        
            

    def run(self, xml_path, image_shape=None):
        gt_data = PageData(xml_path)
        gt_data.parse()

        if image_shape is None:
            image_shape = gt_data.get_size()

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
    
    
    
    

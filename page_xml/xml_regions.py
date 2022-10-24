import argparse
from typing import Optional

class XMLRegions:
    """
    Base for Methods that need to load XML regions
    """
    def __init__(self,
                 mode: str,
                 line_width: Optional[int]=None,
                 line_color: Optional[int]=None,
                 regions: Optional[list[str]]=None,
                 merge_regions: Optional[list[str]]=None,
                 region_type: Optional[list[str]]=None) -> None:
        self.mode = mode
        if self.mode == "baseline":
            assert line_width is not None
            assert line_color is not None

            self.line_width = line_width
            self.line_color = line_color

        elif self.mode == "region":
            assert regions is not None
            assert merge_regions is not None
            # assert region_type is not None
            
            # regions: list of type names (required for lookup)
            # merge_regions: regions to be merged. r1:r2,r3  -> r2 and r3 become region r1
            # region_type: type per_region. t1:r1,r2  -> r1 and r2 become type t1
            self._regions: list[str] = regions
            self._merge_regions: Optional[list[str]] = merge_regions
            self._region_type: Optional[list] = region_type

            self.region_classes = self._build_class_regions()
            self.region_types = self._build_region_types()
            self.merged_regions = self._build_merged_regions()
            self.merge_classes()
        else:
            raise NotImplementedError
    
    #REVIEW is this the best place for this
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        # HACK hardcoded regions if none are given
        republic_regions = ["marginalia", "page-number", "resolution", "date",
                            "index", "attendance", "Resumption", "resumption", "Insertion", "insertion"]
        republic_merge_regions = ["resolution:Resumption,resumption,Insertion,insertion"]
        parser = argparse.ArgumentParser(add_help=False)
        
        region_args = parser.add_argument_group("regions")
        region_args.add_argument(
        "--regions",
        default=republic_regions,
        nargs="+",
        type=str,
        help="""List of regions to be extracted. 
                            Format: --regions r1 r2 r3 ...""",
        )
        region_args.add_argument(
            "--merge_regions",
            default=republic_merge_regions,
            nargs="?",
            const=[],
            type=str,
            help="""Merge regions on PAGE file into a single one.
                                Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                                will be merged into r1 and r5 into r4""",
        )
        region_args.add_argument(
            "--region_type",
            default=None,
            nargs="+",
            type=str,
            help="""Type of region on PAGE file.
                                Format --region_type t1:r1,r3 t2:r5, then type t1
                                will assigned to regions r1 and r3 and type t2 to
                                r5 and so on...""",
        )
        return parser
        
    def merge_classes(self) -> None:
        if self.merged_regions is None:
            return
        if len(self.merged_regions) == 0:
            return
        
        for i, region in enumerate(self.get_regions()):
            self.region_classes[region] = i
        
        for parent, childs in self.merged_regions.items():
            for child in childs:
                self.region_classes[child] = self.region_classes[parent]

    def _build_class_regions(self) -> dict[str, int]:
        """given a list of regions assign a equaly separated class to each one"""

        class_dic = {}

        for c, r in enumerate(self._regions):
            class_dic[r] = c + 1
        return class_dic

    def _build_merged_regions(self) -> Optional[dict[str, str]]:
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
                
        seen_childs = set()
        for childs in to_merge.values():
            for child in childs:
                if child in seen_childs:
                    raise ValueError(f"Found duplicates with {child}")
                if child in to_merge.keys():
                    raise ValueError(f"Found a loop with {child}")
                seen_childs.add(child)

        return to_merge

    def _build_region_types(self) -> dict[str ,str]:
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
    
    def get_regions(self) -> list[str]:    
        remaining_regions = ["background"]
        if self.mode == 'region':
            assert self.merged_regions is not None
            
            removed_regions = set()
            for values in self.merged_regions.values():
                removed_regions = removed_regions.union(set(values))
            remaining_regions.extend(region for region in self._regions if not region in removed_regions)
        else:
            remaining_regions.extend(["baseline"])
        
        return remaining_regions
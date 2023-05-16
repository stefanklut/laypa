import argparse
from typing import Optional

class XMLRegions:
    """
    Base for Methods that need to load XML regions
    """
    def __init__(self,
                 mode: str,
                 line_width: Optional[int]=None,
                 regions: Optional[list[str]]=None,
                 merge_regions: Optional[list[str]]=None,
                 region_type: Optional[list[str]]=None) -> None:
        """
        Base for Methods that need to load XML regions. If type is region specify the region variables otherwise line variables

        Args:
            mode (str): mode of the region type
            line_width (Optional[int], optional): width of line. Defaults to None.
            line_color (Optional[int], optional): value of line (when only one line type exists). Defaults to None.
            regions (Optional[list[str]], optional): list of regions to extract from pageXML. Defaults to None.
            merge_regions (Optional[list[str]], optional): list of region to merge into one. Defaults to None.
            region_type (Optional[list[str]], optional): type of region for each region. Defaults to None.

        Raises:
            NotImplementedError: mode is not known
        """
        self.mode = mode
        if self.mode in ["baseline", "start", "end", "separator", "baseline_separator"]:
            assert line_width is not None

            self.line_width = line_width

        elif self.mode == "region":
            assert regions is not None
            # assert merge_regions is not None
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
    
    #REVIEW is this the best place for this
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """
        Return argparser that has the arguments required for the pageXML regions.
        
        use like this: parser = argparse.ArgumentParser(parents=[XMLConverter.get_parser()])

        Returns:
            argparse.ArgumentParser: the argparser for regions
        """
        # HACK hardcoded regions if none are given
        republic_regions = ["marginalia", "page-number", "resolution", "date",
                            "index", "attendance", "Resumption", "resumption", 
                            "Insertion", "insertion"]
        republic_merge_regions = ["resolution:Resumption,resumption,Insertion,insertion"]
        parser = argparse.ArgumentParser(add_help=False)
        
        region_args = parser.add_argument_group("regions")
        
        region_args.add_argument(
            "-m", "--mode",
            default="region",
            choices=["baseline", "region", "start", "end", "separator", "baseline_separator"], 
            type=str,
            help="Output mode"
        )

        region_args.add_argument(
            "-w", "--line_width",
            default=5,
            type=int,
            help="Used line width"
        )
        
        region_args.add_argument(
            "--regions",
            default=republic_regions,
            nargs="+",
            type=str,
            help="""List of regions to be extracted. 
                                Format: --regions r1 r2 r3 ..."""
        )
        region_args.add_argument(
            "--merge_regions",
            default=republic_merge_regions,
            nargs="?",
            const=[],
            type=str,
            help="""Merge regions on PAGE file into a single one.
                                Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                                will be merged into r1 and r5 into r4"""
        )
        region_args.add_argument(
            "--region_type",
            default=None,
            nargs="+",
            type=str,
            help="""Type of region on PAGE file.
                                Format --region_type t1:r1,r3 t2:r5, then type t1
                                will assigned to regions r1 and r3 and type t2 to
                                r5 and so on..."""
        )
        return parser
        
    def merge_classes(self) -> None:
        """
        Merge the classes defined by the merge regions
        """
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
        """
        Given a list of regions assign a equally separated class to each one
        
        Returns:
            dict[str, str]: keys are the class name, values are the class number
        """

        class_dic = {}

        for c, r in enumerate(self._regions):
            class_dic[r] = c + 1
        return class_dic

    def _build_merged_regions(self) -> dict[str, str]:
        """
        Build dict of regions to be merged into a single class

        Raises:
            argparse.ArgumentTypeError: the command line argument was not given in the right format
            ValueError: found merging into multiple classes (example a:c,b:c)
            ValueError: found merging loop (example: a:b,b:a)

        Returns:
            dict[str, str]: keys are the target class, values are the class to be merged
        """
        if self._merge_regions is None or len(self._merge_regions) == 0:
            return {}
        to_merge = {}
        for c in self._merge_regions:
            parent, childs = c.split(":")
            if parent in self._regions:
                to_merge[parent] = childs.split(",")
            else:
                raise argparse.ArgumentTypeError(
                        f"Malformed argument {c}\nRegion \"{parent}\" to merge is not defined as region"
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
        """ 
        Build a dictionary of regions and their respective type. If none are given, all regions are of type TextRegion.

        Raises:
            argparse.ArgumentTypeError: the command line argument was not given in the right format

        Returns:
            dict[str, str]: Mapping from region to region type
        """
        reg_type = {"full_page": "TextRegion"}
        if self._region_type is None or len(self._region_type) == 0:
            for reg in self._regions:
                reg_type[reg] = "TextRegion"
            return reg_type
        msg = ""
        for c in self._region_type:
            parent, childs = c.split(":")
            regs = childs.split(",")
            for reg in regs:
                if reg in self._regions:
                    reg_type[reg] = parent
                else:
                    raise argparse.ArgumentTypeError(
                        f"Malformed argument {c}\nCannot assign region \"{reg}\" to any type. {reg} not defined as region"
                    )
        return reg_type
    
    def get_regions(self) -> list[str]:
        """
        Return what are currently the regions used. Getting all regions and merging the merged regions.
        Always including a background class as class 0

        Raises:
            NotImplementedError: the mode is not implemented

        Returns:
            list[str]: the names of all the classes currently used
        """
        remaining_regions = ["background"]
        if self.mode == 'region':
            removed_regions = set()
            if self.merged_regions is not None:
                for values in self.merged_regions.values():
                    removed_regions = removed_regions.union(set(values))
            remaining_regions.extend(region for region in self._regions if not region in removed_regions)
        elif self.mode == "baseline":
            remaining_regions.extend(["baseline"])
        elif self.mode == "start":
            remaining_regions.extend(["start"])
        elif self.mode == "end":
            remaining_regions.extend(["end"])
        elif self.mode == "separator":
            remaining_regions.extend(["separator"])
        elif self.mode == "baseline_separator":
            remaining_regions.extend(["baseline","separator"])
        elif self.mode == "text_line":
            remaining_regions.extend('["text_line]')
        else:
            raise NotImplementedError
        
        return remaining_regions
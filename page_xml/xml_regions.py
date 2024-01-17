import argparse
from typing import Optional


class XMLRegions:
    """
    Base for Methods that need to load XML regions
    """

    def __init__(
        self,
        mode: str,
        line_width: Optional[int] = None,
        regions: Optional[list[str]] = None,
        merge_regions: Optional[list[str]] = None,
        region_type: Optional[list[str]] = None,
    ) -> None:
        """
        Base for Methods that need to load XML regions. If type is region specify the region variables otherwise line variables

        Args:
            mode (str): mode of the region type
            line_width (Optional[int], optional): width of line. Defaults to None.
            regions (Optional[list[str]], optional): list of regions to extract from pageXML. Defaults to None.
            merge_regions (Optional[list[str]], optional): list of region to merge into one. Defaults to None.
            region_types (Optional[list[str]], optional): type of region for each region. Defaults to None.
        """
        self.mode = mode
        if self.mode == "region":
            assert regions is not None

            self._regions = []
            self._merge_regions: dict[str, str] = {}
            self._region_type: dict[str, str] = {}

            self._regions_internal = []
            self._merge_regions_internal = None
            self._region_type_internal = None

            # regions: list of type names (required for lookup)
            # merge_regions: regions to be merged. r1:r2,r3  -> r2 and r3 become region r1
            # region_type: type per_region. t1:r1,r2  -> r1 and r2 become type t1
            self.regions = regions
            self.region_types = region_type
            self.merged_regions = merge_regions
        else:
            assert line_width is not None

            self._regions = self._build_regions()
            self.line_width = line_width

    # REVIEW is this the best place for this
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """
        Return argparser that has the arguments required for the pageXML regions.

        use like this: parser = argparse.ArgumentParser(parents=[XMLConverter.get_parser()])

        Returns:
            argparse.ArgumentParser: the argparser for regions
        """
        # HACK hardcoded regions if none are given
        republic_regions = [
            "marginalia",
            "page-number",
            "resolution",
            "date",
            "index",
            "attendance",
            "Resumption",
            "resumption",
            "Insertion",
            "insertion",
        ]
        republic_merge_regions = ["resolution:Resumption,resumption,Insertion,insertion"]
        parser = argparse.ArgumentParser(add_help=False)

        region_args = parser.add_argument_group("Regions")

        region_args.add_argument(
            "-m",
            "--mode",
            default="region",
            choices=["baseline", "region", "start", "end", "separator", "baseline_separator"],
            type=str,
            help="Output mode",
        )

        region_args.add_argument("-w", "--line_width", default=5, type=int, help="Used line width")

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
        if self._merge_regions_internal is None or len(self._merge_regions_internal) == 0:
            return {}
        to_merge = {}
        for c in self._merge_regions_internal:
            parent, childs = c.split(":")
            if parent in self._regions_internal:
                to_merge[parent] = childs.split(",")
            else:
                raise argparse.ArgumentTypeError(f'Malformed argument {c}\nRegion "{parent}" to merge is not defined as region')

        seen_childs = set()
        for childs in to_merge.values():
            for child in childs:
                if child in seen_childs:
                    raise ValueError(f"Found duplicates with {child}")
                if child in to_merge.keys():
                    raise ValueError(f"Found a loop with {child}")
                seen_childs.add(child)

        return to_merge

    def _build_region_types(self) -> dict[str, str]:
        """
        Build a dictionary of regions and their respective type. If none are given, all regions are of type TextRegion.

        Raises:
            argparse.ArgumentTypeError: the command line argument was not given in the right format

        Returns:
            dict[str, str]: Mapping from region to region type
        """
        region_types = {"full_page": "TextRegion"}
        if self._region_type_internal is None or len(self._region_type_internal) == 0:
            for region in self._regions_internal:
                region_types[region] = "TextRegion"
            return region_types

        for c in self._region_type_internal:
            region_type, childs = c.split(":")
            regions = childs.split(",")
            for region in regions:
                if region in self._regions_internal:
                    region_types[region] = region_type
                else:
                    raise argparse.ArgumentTypeError(
                        f'Malformed argument {c}\nCannot assign region "{region}" to any type. {region} not defined as region'
                    )
        for region in self._regions_internal:
            if region not in region_types.keys():
                region_types[region] = "TextRegion"
        return region_types

    def _build_region_classes(self) -> dict[str, int]:
        region_classes = {region: i for i, region in enumerate(self.regions)}

        for parent, childs in self.merged_regions.items():
            for child in childs:
                region_classes[child] = region_classes[parent]

        return region_classes

    def _build_regions(self) -> list[str]:
        """
        Return what are currently the regions used. Getting all regions and merging the merged regions.
        Always including a background class as class 0

        Raises:
            NotImplementedError: the mode is not implemented

        Returns:
            list[str]: the names of all the classes currently used
        """
        remaining_regions = ["background"]
        if self.mode == "region":
            removed_regions = set()
            if self.merged_regions is not None:
                for values in self.merged_regions.values():
                    removed_regions = removed_regions.union(set(values))
            remaining_regions.extend(region for region in self._regions_internal if not region in removed_regions)
        elif self.mode == "baseline":
            remaining_regions.extend(["baseline"])
        elif self.mode == "start":
            remaining_regions.extend(["start"])
        elif self.mode == "end":
            remaining_regions.extend(["end"])
        elif self.mode == "separator":
            remaining_regions.extend(["separator"])
        elif self.mode == "baseline_separator":
            remaining_regions.extend(["baseline", "separator"])
        elif self.mode == "text_line":
            remaining_regions.extend(["text_line"])
        elif self.mode == "top_bottom":
            remaining_regions.extend(["top", "bottom"])
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        return remaining_regions

    @property
    def regions(self) -> list[str]:
        """
        Return the regions to be used

        Returns:
            list[str]: the regions to be used
        """
        return self._regions

    @regions.setter
    def regions(self, regions: list[str]) -> None:
        """
        Set the regions to be used

        Args:
            regions (list[str]): the regions to be used
        """
        self._regions_internal = [region for region in regions if region != "background"]
        self._regions = self._build_regions()
        self._merge_regions = self._build_merged_regions()
        self._region_classes = self._build_region_classes()

    @property
    def region_classes(self) -> dict[str, int]:
        """
        Return the region classes

        Returns:
            dict[str, int]: Mapping from region to class
        """

        return self._region_classes

    @property
    def region_types(self) -> dict[str, str]:
        """
        Return the region types

        Returns:
            Optional[dict[str, str]]: Mapping from region to region type
        """
        return self._region_type

    @region_types.setter
    def region_types(self, region_types: Optional[list[str]]) -> None:
        """
        Set the region types

        Args:
            region_types (dict[str, str]): Mapping from region to region type
        """
        self._region_type_internal = region_types
        self._region_type = self._build_region_types()

    @property
    def merged_regions(self) -> dict[str, str]:
        """
        Return the merged regions

        Returns:
            Optional[dict[str, str]]: Mapping from region to region type
        """
        return self._merge_regions

    @merged_regions.setter
    def merged_regions(self, merged_regions: Optional[list[str]]) -> None:
        """
        Set the merged regions

        Args:
            merged_regions (Optional[dict[str, str]]): Mapping from region to region type
        """
        self._merge_regions_internal = merged_regions
        self._merge_regions = self._build_merged_regions()
        self._regions = self._build_regions()
        self._region_classes = self._build_region_classes()

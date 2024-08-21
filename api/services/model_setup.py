# Imports

# > Standard Libraries
from pathlib import Path
from typing import Optional
import logging

# > Project Libraries
from main import setup_cfg
from page_xml.output_pageXML import OutputPageXML
from page_xml.xml_regions import XMLRegions
from run import Predictor
from api.setup.initialization import DummyArgs


class PredictorGenPageWrapper:
    """
    Wrapper around the page generation code
    """

    def __init__(self, model_base_path: Path) -> None:
        self.model_name: Optional[str] = None
        self.predictor: Optional[Predictor] = None
        self.gen_page: Optional[OutputPageXML] = None
        self.logger = logging.getLogger(__name__)
        self.model_base_path = model_base_path

    def setup_model(self, model_name: str, args: DummyArgs):
        """
        Create the model and post-processing code

        Args:
            model_name (str): Model name, used to determine what model to load from models present in base path
            args (DummyArgs): Dummy version of command line arguments, to set up config
        """
        if (
            model_name is not None
            and self.predictor is not None
            and self.gen_page is not None
            and model_name == self.model_name
        ):
            return

        self.model_name = model_name
        model_path = self.model_base_path.joinpath(self.model_name)
        config_path = model_path.joinpath("config.yaml")
        if not config_path.is_file():
            raise FileNotFoundError(f"config.yaml not found in {model_path}")
        weights_paths = list(model_path.glob("*.pth"))
        if len(weights_paths) < 1 or not weights_paths[0].is_file():
            raise FileNotFoundError(
                f"No valid .pth files found in {model_path}")
        if len(weights_paths) > 1:
            self.logger.warning(
                f"Found multiple .pth files. Using first {weights_paths[0]}")
        args.config = str(config_path)
        args.opts = ["TEST.WEIGHTS", str(weights_paths[0])]

        cfg = setup_cfg(args)
        xml_regions = XMLRegions(cfg)
        self.gen_page = OutputPageXML(
            xml_regions=xml_regions, output_dir=None, cfg=cfg, whitelist={}
        )

        self.predictor = Predictor(cfg=cfg)

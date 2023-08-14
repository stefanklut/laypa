import os
from typing import Optional, Sequence

from pathlib import Path

from detectron2.config import CfgNode

from datasets import dataset
from datasets.preprocess import Preprocess
from utils.input_utils import clean_input_paths, get_file_paths
from page_xml.xml_converter import XMLConverter

def preprocess_datasets(cfg: CfgNode, 
                        train: Optional[str | Path | Sequence[str|Path]], 
                        val: Optional[str | Path | Sequence[str|Path]], 
                        output_dir: str | Path,
                        save_image_locations: bool = True):
    """
    Preprocess the dataset(s). Converts ground truth pageXML to label masks for training

    Args:
        cfg (CfgNode): config
        train (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the training images
        val (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the validation images
        output_dir (str | Path): path to output dir, where the processed data will be saved
        save_image_locations (bool): flag to save processed image locations (for retraining)
        
    Raises:
        FileNotFoundError: a training dir/txt does not exist
        FileNotFoundError: a validation dir/txt does not exist
        FileNotFoundError: the output dir does not exist
    """
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output Folder not found: {output_dir} does not exist")
    
    xml_converter = XMLConverter(
        mode=cfg.MODEL.MODE,
        line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
        regions=cfg.PREPROCESS.REGION.REGIONS,
        merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
        region_type=cfg.PREPROCESS.REGION.REGION_TYPE
    )
    
    assert (n_regions := len(xml_converter.get_regions())) == (n_classes := cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES), \
        f"Number of specified regions ({n_regions}) does not match the number of specified classes ({n_classes})"
    
    process = Preprocess(
        input_paths=None,
        output_dir=None,
        resize_mode=cfg.PREPROCESS.RESIZE.RESIZE_MODE,
        resize_sampling=cfg.PREPROCESS.RESIZE.RESIZE_SAMPLING,
        min_size=cfg.PREPROCESS.RESIZE.MIN_SIZE,
        max_size=cfg.PREPROCESS.RESIZE.MAX_SIZE,
        xml_converter=xml_converter,
        disable_check=cfg.PREPROCESS.DISABLE_CHECK,
        overwrite=cfg.PREPROCESS.OVERWRITE
    )
    
    
    train_output_dir = None
    if train is not None:
        train = clean_input_paths(train)
        if not all(missing := path.exists() for path in train):
            raise FileNotFoundError(f"Train File/Folder not found: {missing} does not exist")
        
        train_output_dir = output_dir.joinpath('train')
        process.set_input_paths(train)
        process.set_output_dir(train_output_dir)
        train_image_paths = get_file_paths(train, process.image_formats)
        process.run()
        
        if save_image_locations:
            #Saving the images used to a txt file
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            train_image_output_path = Path(cfg.OUTPUT_DIR).joinpath("training_images.txt")
        
            with open(train_image_output_path, mode="w") as f:
                for path in train_image_paths:
                    f.write(f"{path}\n")
    
    
    val_output_dir = None
    if val is not None:
        val = clean_input_paths(val)
        if not all((missing := path).exists() for path in val):
            raise FileNotFoundError(f"Validation File/Folder not found: {missing} does not exist")
        
        val_output_dir = output_dir.joinpath('val')
        process.set_input_paths(val)
        process.set_output_dir(val_output_dir)
        val_image_paths = get_file_paths(val, process.image_formats)
        process.run()
        
        if save_image_locations:
            #Saving the images used to a txt file
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            val_image_output_path = Path(cfg.OUTPUT_DIR).joinpath("validation_images.txt")
                
            with open(val_image_output_path, mode="w") as f:
                for path in val_image_paths:
                    f.write(f"{path}\n")
    
    dataset.register_datasets(train_output_dir, 
                              val_output_dir, 
                              train_name="train", 
                              val_name="val")
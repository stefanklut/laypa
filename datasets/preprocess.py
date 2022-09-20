# Taken from P2PaLA

from dataclasses import dataclass
import os
import glob
import logging
import errno
import string
import random
import math
import sys

import numpy as np
import cv2
from shapely.geometry import LineString

import pickle

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from page_xml.xmlPAGE import pageData
from utils import polyapprox as pa


class PreProcess:
    """
    Preprocess files so that regions correspond to image coordinates
    """

    def __init__(self, data_pointer, out_folder, opts, build_labels=True, logger=None):
        """ function to proces all data into a htr dataset"""
        self.logger = logging.getLogger(__name__) if logger == None else logger
        # --- file formats from opencv imread supported formats
        # --- any issue see: https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imread
        self.formats = ["tif", "tiff", "png", "jpg", "jpeg", "JPG", "bmp"]
        self.data_pointer = data_pointer
        self.out_folder = out_folder
        self.build_labels = build_labels
        self.opts = opts
        self.do_class = opts.do_class
        self.line_color = 1 if opts.do_class else opts.line_color
        self.hyp_xml_list = []
        self.validValues = string.ascii_uppercase + string.ascii_lowercase + string.digits
        if self.opts.out_mode == "L":
            self.th_span = 64
        else:
            if len(self.opts.regions_colors.keys()) > 1:
                self.th_span = (
                    self.opts.regions_colors[list(self.opts.regions_colors.keys())[1]]
                    - self.opts.regions_colors[list(self.opts.regions_colors.keys())[0]]
                ) / 2
            else:
                self.th_span = 64

    def pre_process(self):
        """
        """
        # --- Create output folder if not exist
        if not os.path.exists(self.out_folder):
            self.logger.debug("Creating {} folder...".format(self.out_folder))
            os.makedirs(self.out_folder)
            
        self.img_paths = []
        for ext in self.formats:
            self.img_paths.extend(glob.glob(self.data_pointer + "/*." + ext))
            
        img_ids = [os.path.splitext(os.path.basename(x))[0] for x in self.img_paths]
        self.img_data = dict(zip(img_ids, self.img_paths))

        processed_data = []
        
        for i in self.img_paths:
            processed_data.append(
                _processData(
                    (
                        i,
                        self.opts.img_size,
                        self.out_folder,
                        self.opts.regions_colors,
                        self.opts.line_width,
                        self.line_color,
                        self.build_labels,
                        self.opts.out_mode,
                        list(set(self.opts.region_types.values())),
                    )
                )
            )
        processed_data = np.array(processed_data)
        np.savetxt(self.out_folder + "/img.lst", processed_data[:, 0], fmt="%s")
        np.savetxt(self.out_folder + "/label.lst", processed_data[:, 1], fmt="%s")
        np.savetxt(self.out_folder + "/out_size.lst", processed_data[:, 3], fmt="%s")
        if self.build_labels:
            self.label_list = self.out_folder + "/label.lst"
        if self.build_labels:
            self.gt_xml_list = processed_data[:, 2]
        self.out_size_list = processed_data[:, 3]
        if self.build_labels:
            self.gt_xml_list.sort()
        self.img_list = self.out_folder + "/img.lst"
        
# ---- misc functions to this class


def _processData(params):
    """
    Resize image and extract mask from PAGE file 
    """
    (
        img_path,
        out_size,
        out_folder,
        classes,
        line_width,
        line_color,
        build_labels,
        ext_mode,
        node_types,
    ) = params
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img_dir = os.path.dirname(img_path)

    img_data = cv2.imread(img_path)
    # --- resize image

    origheight, origwidth, channels = img_data.shape
    counter = 1
    height = math.ceil(origheight/(256*counter)) * 256
    width = math.ceil(origwidth/(256*counter)) * 256
    while height*width>2048*2048:
        height = math.ceil(origheight/(256*counter)) * 256
        width = math.ceil(origwidth/(256*counter)) * 256
        counter += 1

    out_size = np.array([height, width])
    print(out_size)
    res_img = cv2.resize(
        img_data, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC
    )
    new_img_path = os.path.join(out_folder, img_id + ".png")
    print(new_img_path)
    cv2.imwrite(new_img_path, res_img)
    # --- get label
    if build_labels:
        if os.path.isfile(img_dir + "/page/" + img_id + ".xml"):
            xml_path = img_dir + "/page/" + img_id + ".xml"
        else:
            # logger.critical('No xml found for file {}'.format(img_path))
            # --- TODO move to logger
            print("No xml found for file {}".format(img_path))
            raise FileNotFoundError('No xml found for file {}'.format(img_path))
        gt_data = pageData(xml_path)
        gt_data.parse()
        # --- build lines mask
        if ext_mode != "R":
            lin_mask = gt_data.build_baseline_mask(out_size, line_color, line_width)
        # --- buid regions mask
        if ext_mode == "LR":
            reg_mask = gt_data.build_mask(out_size, node_types, classes)
            label = np.array((lin_mask, reg_mask))
        elif ext_mode == "R":
            label = gt_data.build_mask(out_size, node_types, classes)
        else:
            label = lin_mask

        new_label_path = os.path.join(out_folder, img_id + ".pickle")
        with  open(new_label_path, "wb") as fh:
            pickle.dump(label, fh, -1)
        
        # TODO Should this be used instead?
        # new_label_image_path = os.path.join(out_folder, img_id + ".png")
        # cv2.imwrite(new_label_image_path, label)
            
        return (new_img_path, new_label_path, xml_path,out_size)
    return (new_img_path, None, None,out_size)

if __name__ == "__main__":
    dataset_folder = "/home/stefan/Documents/datasets/cBAD/READ-ICDAR2019-cBAD-dataset-blind/train"
    output_folder = "/tmp/output"
    @dataclass
    class Opts:
        do_class = True
        out_mode = "L"
        regions_colors = 10
        line_width = 5
        img_size = [1024,1024]
        region_types = {"text": "header"}
        
    opts = Opts()

    pre = PreProcess(dataset_folder, output_folder, opts)
    pre.pre_process()

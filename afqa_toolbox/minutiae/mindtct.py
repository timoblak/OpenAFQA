import cv2
import numpy as np
import os
import subprocess
from afqa_toolbox.tools import resized, normed, create_minutiae_record, visualize_minutiae


class MINDTCTWrapper:
    def __init__(self, path_binary, temp_location=None):
        self.path_binary = path_binary
        if temp_location is None:
            print("WARMING: Location for temporary files temp_location not set in MINDTCTWrapper. "
                  "The location was set to current directory ./")
            temp_location = "./"

        temp_location = temp_location + "/" if temp_location[-1] != "/" else temp_location
        self.temp_location = temp_location
        self.generated_files = [".brw", ".dm", ".hcm", ".lcm", ".lfm", ".min", ".qm", ".xyt"]

    def cleanup(self):
        os.remove(self.temp_location + "tmp.png")
        for file in self.generated_files:
            os.remove(self.temp_location + file)

    def read_map(self, filename):
        with open(self.temp_location + filename, "r") as handle:
            content = handle.readlines()
        h, w = len(content), len(content[0].strip().split())
        new_map = np.zeros(shape=(h, w), dtype=int)
        for i, line in enumerate(content):
            new_map[i, :] = [int(x) for x in line.strip().split()]
        return new_map

    def read_minutiae(self):
        with open(self.temp_location + ".min", "r") as handle:
            content = handle.readlines()
        w, h = content[0].strip().split()[2:4]

        minutiae_list = []
        for i, line in enumerate(content):
            if i < 4:
                continue
            results = [x.strip() for x in line.strip().split(":")]
            x, y = [int(x) for x in results[1].split(",")]
            angle_raw = int(results[2])
            angle = ((32 - angle_raw) + 8) % 32
            angle = (angle / 32) * 256

            quality = float(results[3]) * 100
            type = 1 if results[4] == "RIG" else 2
            minutiae_list.append((x, y, type, angle, quality))

        return create_minutiae_record((h, w), minutiae_list)

    def bin_wrapper(self, img, contrast_enhancement=True, clean=True, extended_data=False):
        """
        Wrapper function which uses the MINDTCT binary.
        The function saves a temporary .png, the binary is then called,
        which produces a result files in the given temp_location directory.
        The files are then read and parsed.

        :param img: grayscale image of a fingerprint in 500 ppi
        :param contrast_enhancement: optional flat for
        :param clean: delete temporary files after processing
        :param extended_data: option whether to also read extended data (quality, direction, low-contrast, high-curvature and low-flow maps)
        :return: FMD dict
        """
        if self.path_binary is None:
            print("WARNING: The minutiae couldn't be determined. No path to MINDTCT binary was specified.")
            return None
        image_filename = self.temp_location + "tmp.png"

        # Save temporary image to pgm format
        cv2.imwrite(image_filename, img)

        # Run MINDTCT on saved image
        opt_flag = ""
        if contrast_enhancement:
            opt_flag = "-b"

        #proc = subprocess.Popen([self.path_binary, opt_flag, image_filename, self.temp_location], stdout=subprocess.PIPE, shell=True)
        subprocess.call(" ".join([self.path_binary, opt_flag, image_filename, self.temp_location]), shell=True)
        #print(proc.args)
        #(out, err) = proc.communicate()
        #print(out, err)

        minutiae_data = self.read_minutiae()
        if extended_data:
            direction_map = self.read_map(".dm")
            high_curvature_map = self.read_map(".hcm")
            low_contrast_map = self.read_map(".lcm")
            low_flow_map = self.read_map(".lfm")
            quality_map = self.read_map(".qm")
            if clean:
                self.cleanup()
            return minutiae_data, direction_map, high_curvature_map, low_contrast_map, low_flow_map, quality_map

        if clean:
            self.cleanup()
        return minutiae_data

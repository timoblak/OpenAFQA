import ctypes
import cv2
import numpy as np
from afqa_toolbox.tools.template import *
import subprocess
import os

# Windows 
#FJFX_SAMPLE_BINARY = "../FingerJetFXOSE/FingerJetFXOSE/dist/Windows64/Release/fjfxSample.exe"
#FJFX_LIB_FILENAME = "../FingerJetFXOSE/FingerJetFXOSE/dist/Windows64/Release/FJFX.dll"

# Linux 
#FJFX_SAMPLE_BINARY = "../FingerJetFXOSE/FingerJetFXOSE/dist/Linux-x86_64/x64/fjfxSample"
#FJFX_LIB_FILENAME = "../FingerJetFXOSE/FingerJetFXOSE/dist/Linux-x86_64/x64/libFJFX.so"


def error_code_strings(code):
    if code == 1:
        return "Failed. Input image size was too large or too small."
    elif code == 2:
        return "Failed. Unknown error."
    elif code == 3:
        return "Failed. No fingerprint detected in input image."
    elif code == 7:
        return "Failed. Invalid output record type - only ANSI ... or ISO/IEC ..."
    elif code == 8:
        return "Failed. Output buffer too small."
    else:
        return "Failed. Undefined error code."


class FJFXWrapper:
    FJFX_FMD_BUFFER_SIZE = 34 + 256 * 6
    FJFX_FMD_ISO_19794_2_2005 = 0x01010001
    FJFX_FMD_ANSI_378_2004 = 0x001B0201

    def __init__(self, path_library=None, path_binary=None):
        self.path_library = path_library
        self.path_binary = path_binary

    def bin_wrapper(self, img, temp_location="./", clean=True):
        """
        Wrapper function which uses the example binary (fjfxSample).
        The function saves a temporary .pgm (portable graymap), the binary is then called,
        which produces a template file .ist. The template is then read back.

        :param img: grayscale image of a fingerprint in 500 ppi
        :param temp_location: the location where temporary files will be saved
        :param clean: delete temporary files after processing
        :return: FMD dict
        """
        if self.path_binary is None:
            print("WARNING: The minutiae couldn't be determined. No path to FJFX binary was specified.")
            return None
        template_filename = temp_location + "tmp.ist"
        image_filename = temp_location + "tmp_img.pgm"

        # Save temporary image to pgm format
        cv2.imwrite(image_filename, img)

        # Run FJFX on saved image
        proc = subprocess.Popen([self.path_binary, image_filename, template_filename], stdout=subprocess.PIPE, shell=True)

        (out, err) = proc.communicate()
        if out.decode("utf-8").strip() == "Failed feature extraction" or err:
            return None

        fl = read_minutiae_file(template_filename)
        if clean:
            os.remove(template_filename)
            os.remove(image_filename)
        return fl

    def lib_wrapper(self, img, ppi):
        """
        Ctypes wrapper function for the fjfx.dll library.
        The function fjfx_create_fmd_from_raw is called and Fingerprint Minutiae Data (FMD) is retrieved.

        :param img: grayscale image
        :param ppi: ppi of the input fingerprint image
        :return: FMD dict
        """
        if self.path_library is None:
            print("WARNING: The minutiae couldn't be determined. No path to FJFX library was specified.")
            return None

        fjfx_lib = ctypes.CDLL(self.path_library)
        fjfx_create_fmd_from_raw = fjfx_lib.fjfx_create_fmd_from_raw
        fjfx_create_fmd_from_raw.restype = ctypes.c_int
        fjfx_create_fmd_from_raw.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ushort,
            ctypes.c_ushort,
            ctypes.c_ushort,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint)
        ]

        fmd = np.zeros(shape=(self.FJFX_FMD_BUFFER_SIZE,), dtype=np.uint8)

        # create a pointer object to access the value later
        pointer = ctypes.POINTER(ctypes.c_uint)
        buffer_size_ptr = pointer(ctypes.c_uint(self.FJFX_FMD_BUFFER_SIZE))

        ret_val = fjfx_create_fmd_from_raw(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ppi,
            img.shape[0],
            img.shape[1],
            self.FJFX_FMD_ISO_19794_2_2005,
            fmd.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            buffer_size_ptr
        )

        # Error codes:
        # FJFX_SUCCESS                         (0)     // Extraction succeeded, minutiae data is in output buffer.
        # FJFX_FAIL_IMAGE_SIZE_NOT_SUP         (1)     // Failed. Input image size was too large or too small.
        # FJFX_FAIL_EXTRACTION_UNSPEC          (2)     // Failed. Unknown error.
        # FJFX_FAIL_EXTRACTION_BAD_IMP         (3)     // Failed. No fingerprint detected in input image.
        # FJFX_FAIL_INVALID_OUTPUT_FORMAT      (7)     // Failed. Invalid output record type - only ANSI ... or ISO/IEC ...
        # FJFX_FAIL_OUTPUT_BUFFER_IS_TOO_SMALL (8)     // Failed. Output buffer too small.

        if ret_val > 0:
            print(error_code_strings(ret_val))
            return dummy_minutiae_record()

        # buffer_size now contains the actual size of output buffer
        # crop the final FMD to it's actual size
        buffer_size = buffer_size_ptr.contents.value
        actual_fmd = bytearray(fmd.tobytes())[:buffer_size]

        return parse_minutiae_bytes(actual_fmd)

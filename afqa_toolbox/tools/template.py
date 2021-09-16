import struct


def pop_n(seq, n):
    """
    Pops first n elements from a mutable bytearray
    :param seq: Input bytearray
    :param n: The number of elements to pop
    :return: First n elements of the input bytearray
    """
    tmp = seq[:n]
    del seq[:n]
    return bytes(tmp)


def read_minutiae_file(filename):
    """
    Reads template file and parses to dictionary template
    :param filename: Path to template file
    :return: A template dictionary
    """
    with open(filename, "rb") as f:
        data = bytearray(f.read())

    return parse_minutiae_bytes(data)


def parse_minutiae_bytes(seq):
    """
    Parses binary structure of standard ISO/IEC 19794-2:2005/Cor.1:2009
    For a thorough explanation of template fields check the following unofficial documentation:
    https://templates.machinezoo.com/iso-19794-2-2005
    :param filename: Path to template file
    :return: A template dictionary
    """
    template = dict()
    # Process header (24 Bytes)
    # MAGIC string "FMR\0"
    template["magic"] = pop_n(seq, 4).decode('utf-8')  # Unused
    # VERSION to distinguish from other template formats
    template["version"] = pop_n(seq, 4).decode('utf-8')  # Unused
    # TOTALBYTES total length in bytes
    template["totalbytes"] = struct.unpack(">I", pop_n(seq, 4))[0]

    # DEVSTAMP [top bit ON or OFF] indicates complience with Annex B of this ISO 19794-2 spec
    # DEVID [?] id of fingerprint reader
    # Zeored for open-source implementations
    template["devstamp_devid"] = pop_n(seq, 2).decode('utf-8')  # Unused

    # WIDTH, HEIGHT of image
    template["width"] = struct.unpack(">H", pop_n(seq, 2))[0]
    template["height"] = struct.unpack(">H", pop_n(seq, 2))[0]

    # RESX, RESY [99 - ?] pixels per centimeter; 500 px/i = 179 px/cm
    template["resx"] = struct.unpack(">H", pop_n(seq, 2))[0]
    template["resy"] = struct.unpack(">H", pop_n(seq, 2))[0]

    # FPCOUNT [0 - 176] number of fingerprints
    template["fpcount"] = struct.unpack(">B", pop_n(seq, 1))[0]

    # Check if byte at end of header is empty
    emptybyte = struct.unpack(">B", pop_n(seq, 1))[0]
    assert emptybyte == 0

    # Process Fingerprint (m * (6 + (n*6)) Bytes)
    # m = number of fingerprints, n = number of minutiae
    template["fingerprints"] = []
    for m in range(template["fpcount"]):
        fingerprint = dict()
        # POSITION [0 - 10] id of finger:
        # 0 - unknown, 1-5 right thumb to little, 6-7 left thumb to little
        fingerprint["position"] = struct.unpack(">B", pop_n(seq, 1))[0]

        # Take first four bits for viewoffset and second four for sampletype
        # VIEWOFFSET [0 - 16] if multiple fingerprints with same POSITION are present in template
        # SAMPLETYPE [0 - 3, 8] Impression type:
        # 0 - live plain, 1 - live rolled, 2 - non-live plain, 3 - non-live rolled, 8 - swipe
        viewoffset_sampletype = int.from_bytes(pop_n(seq, 1), "big")
        fingerprint["viewoffset"] = viewoffset_sampletype >> 4
        fingerprint["sampletype"] = viewoffset_sampletype & 0x0F

        # FPQUALITY [0 - 100] quality of fingerprint
        # MINCOUNT [0 - ?] number of minutiae points
        fingerprint["fpquality"] = struct.unpack(">B", pop_n(seq, 1))[0]
        fingerprint["mincount"] = struct.unpack(">B", pop_n(seq, 1))[0]

        fingerprint["minutiae"] = []
        # Process Minutiae (6 Bytes)
        for n in range(fingerprint["mincount"]):
            minutiae = dict()

            # Take first two bits for mintype and next 14 for minx
            # MINTYPE [0 - 2] type of minutiae: 1 - ending, 2 - bifurcation, 0 - other
            # MINX [0 - w-1] y coordinate of minutiea
            mintype_minx = int.from_bytes(pop_n(seq, 2), "big")
            minutiae["mintype"] = mintype_minx >> 14
            minutiae["minx"] = mintype_minx & 0x3FFF

            # MINY [0 - h-1] y coordinate of minutiea
            minutiae["miny"] = int.from_bytes(pop_n(seq, 2), "big")

            # MINANGLE [0 - 255] quantized angle, start is on right, increases counterclockwise
            # MINQUALITY [0 - 100] zero if not reported
            minutiae["minangle"] = struct.unpack(">B", pop_n(seq, 1))[0]
            minutiae["minquality"] = struct.unpack(">B", pop_n(seq, 1))[0]

            fingerprint["minutiae"].append(minutiae)

        # EXTBYTES total size of all extensions
        fingerprint["extbytes"] = struct.unpack(">H", pop_n(seq, 2))[0]

        # Currently only minutiae templates from FJFX are read, which don't use extention data
        # TODO: Implement reading of extension data.

        template["fingerprints"].append(fingerprint)
    return template


def create_minutiae_record(im_shape, minutiae_list):
    """
    Creates a dictionary of minutiae information.
    The structure is similar to the template structure of standard ISO/IEC 19794-2:2005/Cor.1:2009
    :param filename: List of minutiae data in form [(x, y, type, angle, quality), ...]
    :return: A template disctionary
    """
    template = dict()
    # Process header (24 Bytes)
    template["magic"] = None
    template["version"] = None
    template["totalbytes"] = None
    template["devstamp_devid"] = None

    # WIDTH, HEIGHT of image
    template["width"] = im_shape[1]
    template["height"] = im_shape[0]

    # RESX, RESY [99 - ?] pixels per centimeter; 500 px/i = 179 px/cm
    template["resx"] = 179
    template["resy"] = 179

    # FPCOUNT [0 - 176] number of fingerprints
    template["fpcount"] = 1

    # m = number of fingerprints, n = number of minutiae
    template["fingerprints"] = []
    for m in range(template["fpcount"]):
        fingerprint = dict()
        fingerprint["position"] = None
        fingerprint["viewoffset"] = None
        fingerprint["sampletype"] = None
        # FPQUALITY [0 - 100] quality of fingerprint
        fingerprint["fpquality"] = 60
        # MINCOUNT [0 - ?] number of minutiae points
        fingerprint["mincount"] = len(minutiae_list)

        fingerprint["minutiae"] = []
        # Process Minutiae (6 Bytes)
        for minutia in minutiae_list:
            minutiae = dict()
            # MINTYPE [0 - 2] type of minutiae: 1 - ending, 2 - bifurcation
            minutiae["mintype"] = minutia[2]

            # MINX [0 - w-1] y coordinate of minutiea
            minutiae["minx"] = minutia[0]

            # MINY [0 - h-1] y coordinate of minutiea
            minutiae["miny"] = minutia[1]

            # MINANGLE [0 - 255] quantized angle, start is on right, increases counterclockwise
            minutiae["minangle"] = minutia[3]

            # MINQUALITY [0 - 100] zero if not reported
            minutiae["minquality"] = minutia[4]

            fingerprint["minutiae"].append(minutiae)

        fingerprint["extbytes"] = None
        template["fingerprints"].append(fingerprint)
    return template


def dummy_minutiae_record():
    """A dummy fingerprint minutiae record"""
    template = dict()
    # Process header (24 Bytes)
    # MAGIC string "FMR\0"
    template["magic"] = None
    # VERSION to distinguish from other template formats
    template["version"] = None
    # TOTALBYTES total length in bytes
    template["totalbytes"] = None

    # DEVSTAMP [top bit ON or OFF] indicates complience with Annex B of this ISO 19794-2 spec
    # DEVID [?] id of fingerprint reader
    # Zeored for open-source implementations
    template["devstamp_devid"] = None

    # WIDTH, HEIGHT of image
    template["width"] = None
    template["height"] = None

    # RESX, RESY [99 - ?] pixels per centimeter; 500 px/i = 179 px/cm
    template["resx"] = None
    template["resy"] = None

    # FPCOUNT [0 - 176] number of fingerprints
    template["fpcount"] = 0

    template["fingerprints"] = []
    return template
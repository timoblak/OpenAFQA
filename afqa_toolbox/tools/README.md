
## Common tools 

Here are some of the more common helper functions for friction ridge processing.

### Template read & write

Currently we support: 
- Reading from a ISO/IEC 19794-2:2005/Cor.1:2009 minutiae data record. [`read_minutiae_file()`, `parse_minutiae_bytes()`]
- To save minutiae data in a more human readable format, we use a Python dictionary with similar structure [`create_minutiae_record()`]. The dictionary can then be saved in a .json file. 

#### Visualization 

- Visualize an orientation field (angles in radians starting on left, going counterclockwise) [`visualize_orientation_field()`]
- Visualize minutiae data on an image (colored either by quality or type) [`visualize_minutiae()`]

#### Other tools

- Simple image resizing and normalization [`normed()`, `resized()`]
- Generic sliding windows (similar to matlabs blockproc) [`blockproc()`, `blockproc_inplace()`]
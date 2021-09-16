## Segmentation of friction ridge impressions
Included here are various algorithms for fingermark and fingerprint segmentation

### Algorithms

#### Intensity based 

- Based on standard deviation. Image blocks with large std are determined to be foreground. [`std_segmentation()`]
- Threshold of equalized image. Image is first equalized with histogram equalization. Foreground is determined based on given threshold and mask is refined with morphological operations.  [`hist_equalization_segmentation()`]
 
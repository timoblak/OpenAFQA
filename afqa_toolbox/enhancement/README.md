## Enhancement of friction ridge impressions
Included here are various algorithms for fingermark and fingerprint enhancement

### Algorithms

#### Image filters

- Difference of Gaussians. The filter enhances local contrast within the image. [`dog_filter()`]
- Frequency filter. This is a magnitude filter for 2D images which blocks frequencies with lowest k% of frequencies. [`magnitude_filter()`]

#### Oriented Gabor Enhancement 

- Fingerprint enhancement based on oriented Gabor filters, also called [Hongs method](https://doi.org/10.1109/34.709565). [`GaborEnhancement`]
 

###### *TODO*
- Short time Fourier transform ([STFT](https://doi.org/10.1016/j.patcog.2006.05.036)) fingerprint enhancement. 
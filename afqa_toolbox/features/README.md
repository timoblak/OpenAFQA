## Friction ridge image-level feature extractors 
The feature extraction algorithms provided here analyze a local area of the friction ridge impression. 

### Usage
Most of the are implemented as a class, containing 2 methods: (i) One takes as input a whole image and returns a feature map corresponding to the feature. (ii) The other method is a static method and operates on a single local block and returns a single value. This can be used if you want to implement your own sliding window across image data. 

Example: `FeatFDA` is the class for Frequency Domain Analysis and contains methods `fda()`, which processed the whole image, and `fda_block()`, which processes only a local image block.  

Some features (notably NFIQ 2 features) implement a slanted block extraction due to rotating the local image data. 

### Features

##### NFIQ 2 features
We implement a number of NFIQ 2 local feature. For more information about these, see the [NFIQ 2 report](https://www.nist.gov/system/files/documents/2018/11/29/nfiq2_report.pdf).
- Frequency Domain Analysis [`FeatFDA`]
- Local Clarity Score [`FeatLCS`]
- Orientation Certainty Level [`FeatOCL`]
- Orientation Flow [`FeatOFL`]
- Ridge Valley Uniformity [`FeatRVU`]
- Local orientation based on image gradient covariances [`orient()`]   
- Local mean and standard deviation [`FeatStats`]

##### DFIQI features
Another set of local quality features are described by [*Swafford et al*](https://doi.org/10.1016/j.forsciint.2021.110703).
- Acutance [`FeatACUT`]
- Mean Object Width [`FeatMOW`]
- Spatial Frequency [`FeatSF`]
- Signal Percent Pixels Per Grid [`FeatS3PG`]
- Bimodal Separaiton [`FeatSEP`] 

##### Other features
Other common local quality indicators for friction ridge images, including our own additions: 
- Response to oriented Gabor filters [`FeatGabor`]
- Ridge Discontinuity Indicator [`FeatRDI`]

## Minutiae extraction algorithms 

Minutiae extraction is a complex process, particularly for (often noisy) fingermarks. As several open-source minutiae extraction libraries exist, we provide the necessary Python wrapper classes. We also provide a Python implementation of a simple and customizable minutiae extractor.  

All minutiae extraction algorithms result in a Python dictionary, similar in structure to the ISO/IEC 19794-2:2005/Cor.1:2009 minutiae data record specification. For the structure of the dictionary, see [tools](../tools). For a thorough explanation of template fields check the following [unofficial documentation](https://templates.machinezoo.com/iso-19794-2-2005). 

 

### Algorithms
#### MINDTCT [wrapper]

The algorithm was designed by NIST and is used for fingerprint minutiae extraction. The algorithm saw use in the FBI's ULW (Universal Latent Workstation) AFIS software and was used as minutiae extractor in NFIQ. For more information on the algorithm, see the [user guide to NIST Biometric Image Software (NBIS)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51097).

Instructions: 
1. Download and compile NISTs [NBIS software](https://www.nist.gov/itl/iad/image-group/products-and-services/image-group-open-source-server-nigos#Releases) (tested with version 5.0.0). 
2. Initialize the Python wrapper class [`MINDTCTWrapper`] and provide it with the path to the compiled mindtct binary (`mindtct.exe` in Windows and `mindtct` in Linux). 
3. The wrapper class calls the mindtct binary and saves intermediate results in a temporary folder. 
4. The wrapper class can also read generated feature maps (such as direction, orientation, high curvature and low contrast maps, used internally to compute MINDTCT).

#### FingerJetFX OSE [wrapper] 

One of the few publicly available open source implementations, knows for it's robustness and good performance. It was selected as the minutiae extractor of choice for the NIST NFIQ2.0. 

Instructions: 
1. Download and compile the [FingerJetFX source code](https://github.com/FingerJetFXOSE/FingerJetFXOSE).
2. Initialize the Python wrapper class [`FJFXWrapper`] and provide it with the path to the compiled FJFX library or the provided binary example (`fjfxSample.exe`,`FJFX.dll` in Windows or `fjfxSample`,`libFJFX.so` in Linux).
3. Using the binary wrapper [`FJFXWrapper.bin_wrapper()`] writes the results to disk. We recommend using the library wrapper [`FJFXWrapper.lib_wrapper()`], which uses ctypes to call the FJFX minutiae extraction directly from a C function. Calling the library is faster and more customizable. 

#### Minutiae extraction using the Crossing Number (CN) method

We implemented the commonly used Crossing Number (CN) method ([*Kasaei et al.*](https://doi.org/10.1109/TENCON.1997.647317), [*Amengual et al.*](https://doi.org/10.1049/cp:19971021), [*B. M. Mehtre*](https://doi.org/10.1007/BF01211936)), similarly as described by [Raymond Thai](https://www.peterkovesi.com/studentprojects/raymondthai/RaymondThai.pdf). This method does not compute the quality/reliability of infividual minutiae, so a placeholder value of 60 is used instead.

Algorithm outline:
1. The method receives as input a binarized friction ridge image, where the pixels, belonging to ridges, have a value of 1 and the rest have a value of 0 (the included Gabor enhancement can be used to properly binarize the image.)
2. The binarized image is thinned to produce a skeleton image, where all ridges have a width of 1 pixel and are 8-connected. 
3. All pixels, belonging to thinned ridges are analyzed with the CN method to detect minutiae.
4. Each minutia is verified by analyzing its local neighborhood (radius = 5) and tracing each ridge originating from the minutia location. 
5. The detected minutiae are filtered using a distance filter.

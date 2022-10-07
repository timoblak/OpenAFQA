
## Friction ridge impression Quality Assessment methods

A collection of implemented related work and methods, developed by us. 

### Our methods

#### AFQA Ensemble (2022)
Here are the implementations of methods presented by Oblak et al. (Knowledge-based systems, 2022). They include:
- **The classic ensemble** (The classic predictive pipeline - Preprocessing, feature extraction, and a random forest quality regressor, pretrained on SD302 dataset)
- **The deep learning ensemble** (A DenseNet model, pretrained on SD302 dataset)

Both methods also include a pretrained fusion method. The pre-trained models and other external files can be downloaded from this [link](https://unilj-my.sharepoint.com/:u:/g/personal/tim_oblak_fri1_uni-lj_si/EWsIr-hK01NJit-NN8XufwIB3uTDYcz4xBjcQ9rYA_rzHA?e=kOWa2n).
For more information on how to use these methods, see [Toolbox examples](../../toolbox_examples).

If you use our methods in your research, please cite: 
    
    T. Oblak, R. Haraksim, P. Peer, L. Beslay. 
    Fingermark quality assessment framework with classic and deep learning ensemble models. 
    Knowledge-Based Systems, Volume 250, 2022    

    T. Oblak, R. Haraksim, L. Beslay, P. Peer. 
    Fingermark Quality Assessment: An Open-Source Toolbox. 
    In proceedings of the International Conference of the Biometrics Special Interest Group (BIOSIG), pp. 159-170, 2021.
     

### Related work

#### DFIQI 
A minutiae based quality assessment of fingermarks. This is an implementation of paper from by [*Swafford et al*](https://doi.org/10.1016/j.forsciint.2021.110703). For more info, see the original paper. 

#### Gabor quality
A global quality assessment method based on Gabor filters by [*Shen et al*](https://doi.org/10.1007/3-540-45344-X_39). For more info, see the original paper.


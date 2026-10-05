
## Friction ridge impression Quality Assessment methods

A collection of implemented related work and methods, developed by us. 

### Our methods

#### pAFQA (2023)
The probabilistic fingermark quality assessment model, presented by Oblak et al. (Sensors, 2023). A ResNet-34 model, pretrained on SD302 dataset, predicts a probability distribution over 100 quality bins. The final quality value (1-100) is the expected value of the predicted distribution, while its standard deviation can be used as a measure of the model's uncertainty [`PAFQA`]. 
The pre-trained model can be downloaded from this [link](https://unilj-my.sharepoint.com/:u:/g/personal/tim_oblak_fri1_uni-lj_si/IQBny4H7ToQxRZ34RGlFwj-hAWdkypPKVZoIwtRa6yJ6XHI?e=mqYpdF).

    T. Oblak, R. Haraksim, L. Beslay, P. Peer. 
    Probabilistic Fingermark Quality Assessment with Quality Region Localisation. 
    Sensors, Volume 23, Issue 8, 4006, 2023

#### AFQA Ensemble (2022)
Here are the implementations of methods presented by Oblak et al. (Knowledge-based systems, 2022). They include:
- **The classic ensemble** (The classic predictive pipeline - Preprocessing, feature extraction, and a random forest quality regressor, pretrained on SD302 dataset)
- **The deep learning ensemble** (A DenseNet model, pretrained on SD302 dataset)

Both methods also include a pretrained fusion method. The pre-trained models and other external files can be downloaded from this [link](https://unilj-my.sharepoint.com/:u:/g/personal/tim_oblak_fri1_uni-lj_si/EWsIr-hK01NJit-NN8XufwIB3uTDYcz4xBjcQ9rYA_rzHA?e=kOWa2n).
For more information on how to use these methods, see [Toolbox examples](../../toolbox_examples). Note that the pretrained random forest models of the classic ensemble require `scikit-learn<1.3`.

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


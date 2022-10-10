import cv2
from afqa_toolbox.quality import DeepEnsemble, ClassicEnsemble
import time


latent = ""  # Path to latent/fingermark image
image = cv2.imread(latent, 0)

deep_model_path = "external_resources/model_densenet121.pt"
classic_model_path = "external_resources/models_randomforest.pkl"

# For improved performance, the deep models can be executed on the GPU
# device = "cpu"
device = "cuda:0"

deep_q = DeepEnsemble(deep_model_path=deep_model_path, device=device)
t0 = time.time()
ensemble_prediction = deep_q.predict_ensemble(image)
fusion_quality = deep_q.fusion(ensemble_prediction)

print("--- Results deep model ---")
print("Execution time: " + str(time.time() - t0))
print("Ensemble predictions:", ensemble_prediction)
print("Fused quality:", fusion_quality)

# Resizing image to 500 ppi
ppi = int(latent.split("/")[-1].split("_")[6].replace("PPI", ""))
image500 = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)

ensemble_models_path = "external_resources/models_randomforest.pkl"
mindet_lib_path = "external_resources/libFJFX.so"

q = ClassicEnsemble(ensemble_models_path=ensemble_models_path, mindet_lib_path=mindet_lib_path)
t0 = time.time()
feature_vector = q.feature_vector(image500)
ensemble_prediction = q.predict_ensemble(feature_vector)
fusion_quality = q.fusion(ensemble_prediction)

print("--- Results classic model ---")
print("Execution time: " + str(time.time() - t0))
print("Ensemble predictions:", ensemble_prediction)
print("Fused quality:", fusion_quality)

import pickle
import os

model_path = '/mnt/c/Users/sneha/Dengue-prediction/dengue_model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded: {type(model)}")
    print(f"Model attributes: {dir(model)}")
    if hasattr(model, 'feature_names_in_'):
        print(f"feature_names_in_: {model.feature_names_in_}")
    if hasattr(model, 'n_features_in_'):
        print(f"n_features_in_: {model.n_features_in_}")
    if hasattr(model, 'classes_'):
        print(f"classes_: {model.classes_}")
except Exception as e:
    print(f"Error: {e}")

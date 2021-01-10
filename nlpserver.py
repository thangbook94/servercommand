from joblib import dump, load

# load it again
with open('save_model.pkl', 'rb') as fid:
    gnb_loaded = load(fid)
gnb_loaded.predict_proba("đsdsdsdsds")
print(gnb_loaded)
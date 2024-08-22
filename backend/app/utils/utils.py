import os
from .app_path import AppPath

def save_cache(image_name, image_path, predicted_name, predicted_alias, probs, best_prob, pred_id, pred_class):
    cache_path = f'{AppPath.CACHE_DIR}/predicted_cache.csv'
    cache_exists = os.path.isfile(cache_path)
    with open(cache_path, 'a') as f:
        if not cache_exists:
            f.write('Image_name, Image_path, Predicted_name, Predicted_alias, Probs, Best_prob, Predicted_id, Predicted_class\n')
        f.write(f'{image_name},{image_path},{predicted_name},{predicted_alias},{probs},{best_prob},{pred_id},{pred_class}\n')
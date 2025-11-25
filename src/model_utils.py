import pandas as pd

def rate_video(metadata, model):
    row = pd.DataFrame([metadata])
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    predicted_class = classes[proba.argmax()]
    
    trend_score = None
    if 'trending' in classes:
        trend_score = proba[classes.index('trending')] * 100
    
    return predicted_class, trend_score


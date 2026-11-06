import os
import pandas as pd
from prefect import flow, task
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

@task
def load_data():
    # Expects a spotify tracks dataset placed in data/spotify_tracks.csv
    df = pd.read_csv("data/spotify_tracks.csv")
    return df

@task
def clean_data(df):
    features = [
        "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    target = "popularity"
    
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def train_evaluate(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Music_Popularity_Experiment")
    
    with mlflow.start_run(run_name="music_rf_run") as run:
        n_estimators = 100
        max_depth = 10
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(model, "model", registered_model_name="MusicPopularityModel")
    return run.info.run_id

@flow(name="music_pipeline")
def main_flow():
    df = load_data()
    X_train, X_test, y_train, y_test = clean_data(df)
    train_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main_flow()
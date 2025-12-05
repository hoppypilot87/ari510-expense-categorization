import pandas as pd
from model_predict import predict_category_from_vector

def main():
    import os
    input_path = os.environ["INPUT_FILE"]         # Codabench provides this
    output_path = os.environ["PREDICTION_FILE"]   # You must write to this

    X = pd.read_csv(input_path)
    predictions = []

    for _, row in X.iterrows():
        vec = row.values
        pred = predict_category_from_vector(vec)
        predictions.append(pred)

    pd.DataFrame({"category": predictions}).to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

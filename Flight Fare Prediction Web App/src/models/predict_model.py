import pickle


def predict_model(model_path, input_data):
    model = pickle.load(open(model_path, "rb"))
    y_pred = model.predict(input_data)
    return y_pred

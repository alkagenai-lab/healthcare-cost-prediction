import joblib
import pandas as pd

MODEL_PATH = "ml/artifacts/model.pkl"


def test_model_artifacts_exist():
    ...


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample_data = pd.DataFrame(
        [
            {
                "age": 45,
                "bmi": 30.0,
                "children": 2,
                "sex": "male",
                "smoker": "no",
                "region": "northwest",
            }
        ]
    )

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (int, float))
    assert prediction[0] > 0
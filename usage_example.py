import pandas as pd
import numpy as np

from LOCATE_functions import LOCATE_training, LOCATE_predict

if __name__ == '__main__':
    # Generate random data for example
    X_train = np.log(pd.DataFrame(np.random.uniform(0, 1, size=(100, 50)), index=[f"a_{i}" for i in range(100)]) + 0.1)
    X_val = np.log(pd.DataFrame(np.random.uniform(0, 1, size=(30, 50)), index=[f"b_{i}" for i in range(30)]) + 0.1)

    Y_train = np.log(pd.DataFrame(np.random.uniform(0, 1, size=(100, 30)), index=[f"a_{i}" for i in range(100)]) + 0.1)
    Y_val = np.log(pd.DataFrame(np.random.uniform(0, 1, size=(30, 30)), index=[f"b_{i}" for i in range(30)]) + 0.1)

    # Training the model
    model = LOCATE_training(X_train, Y_train, X_val, Y_val)
    # Prediction
    Z_val, n_pred = LOCATE_predict(model, X_val, Y_val.columns)

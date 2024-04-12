from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

def train_models(X, y):
    # Define the pipeline for Logistic Regression
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Define the pipeline for LightGBM
    pipe_lgbm = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier())
    ])

    # List of pipelines for ease of iteration
    pipelines = [pipe_lr, pipe_lgbm]

    # Dictionary of pipelines and classifier types for ease of reference
    pipe_dict = {0: 'Logistic Regression', 1: 'LightGBM'}

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipelines
    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    # Compare accuracies
    for idx, val in enumerate(pipelines):
        print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))
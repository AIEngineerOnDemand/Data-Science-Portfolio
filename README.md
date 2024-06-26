# Data Science Portfolio

This repository showcases my skills and knowledge in Data Science and Machine Learning. It contains various projects and code snippets demonstrating my proficiency in the following libraries and frameworks:

- **Scikit-learn**: A powerful Python library for machine learning. It contains function for regression, classification, clustering, model selection and dimensionality reduction.

- **SciPy**: A Python library used for scientific and technical computing.

- **TensorFlow**: An end-to-end open source platform for machine learning developed by Google Brain Team.

- **Keras**: A user-friendly neural network library written in Python.

- **PyTorch**: An open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.

- **Pandas**: A library providing high-performance, easy-to-use data structures and data analysis tools.

- **NumPy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

- **Plotly**: A graphing library makes interactive, publication-quality graphs online.

- **Seaborn**: A Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

- **Matplotlib**: A plotting library for the Python programming language and its numerical mathematics extension NumPy.

- **NLTK (Natural Language Toolkit)**: A leading platform for building Python programs to work with human language data.

- **Spacy**: A library for advanced Natural Language Processing in Python and Cython.

## Project Structure

This project follows a modular structure, which is organized as follows:

- `data`: Contains raw and processed data. You might want to gitignore this.
- `notebooks`: Jupyter notebooks for exploratory data analysis and reporting. You can put notebooks at the root of this directory, or organize them into subdirectories if the project grows large.
- `src`: Python scripts go here. It is further divided into:
    - `data`: Scripts to download or generate data.
    - `features`: Scripts to turn raw data into features for modeling.
    - `models`: Scripts to train models and then use trained models to make predictions.
    - `visualization`: Scripts to create exploratory and results-oriented visualizations.
- `tests`: Contains unit tests.

In addition to these, the project uses Poetry for package management, which is reflected in the `pyproject.toml` and `poetry.lock` files. The project is also Dockerized, with the `Dockerfile` at the root of the project directory.


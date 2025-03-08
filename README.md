# ⛩️ deeplearning-dojo

Coding exercises for the deeplearning repo.

## 🎉 Setup

> We use `venv`. If you prefer you can use `conda` or `docker`. Dependencies are in `requirements.txt`.

```
git clone https://github.com/vikasraykar/deeplearning-dojo.git
cd deeplearning-dojo

python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Write you solutions in the files provided in the `stubs` folder.

```
cd stubs
ipython
run LinearRegressionNumpy.py
```

```
deactivate
```

## 📚 Documentation

https://vikasraykar.github.io/deeplearning/

## 📁 Folder structure

The repo is organized as follows.

folder | description
:--- | :---
`stubs` | Sample stubs are provided in this folder. YOu task is to write code here.
`tests` | Tests written with [pytest](https://docs.pytest.org/en/latest/).
`solutions` | The complete solutions.

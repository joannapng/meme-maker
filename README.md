Create a new anaconda environment

```
conda create -n project_env python=3.11
```

Activate the environment with

```
conda activate project_env
```

To install the dependences, cd into the project directory and run

```
pip install -r requirements.txt
```

To install **ipywidgets**

```
conda install -n project_env -c conda-forge ipywidgets
```

The jupyter notebook will run in the base environment

```
conda install -n base -c conda-forge widgetsnbextension
```

In the project environment

```
conda install ipykernel
conda install nb_conda
```

Open jupyter notebook in the project_env environment and choose the project_env kernel

TODO
- Make the slider limits depend on the image size
- Add preview to save image page
- Add undo button to filters
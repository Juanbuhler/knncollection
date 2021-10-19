# KNN Collection

A simple image collection that implements semantic similarity, image clustering and KNN classification.

## Getting started

Python 3 is required, replace "pip3" and "python3" accordingly if necessary in your system.

- Install requirements:

`pip install -r requirements.txt`

- Preprocess demo images

A folder with demo images from the [USDA Pomological Watercolors](https://naldc.nal.usda.gov/usda_pomological_watercolor_collection) set is included. You can replace your own images in another folder. The browser will let you select a collection of images from all the collections for which this preprocess has been run.

`python preprocess.py`

The first time this is run, it will download the weights for a Resnet CNN that it used to compute image similarity vectors. Subsequent runs should be faster, although if you have many images, this can take a long time.

- Run Streamlit based browser:

`streamlit run sim_search.py`

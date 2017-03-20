from glob import glob
import pandas as pd
from pandas import DataFrame, Series

def retrieve_data(category):
    paths = glob("training_images/" + category + "/*", recursive=True)
    subcategories = [path.split('/')[-1] for path in paths]
    images = []
    for subcategory in subcategories:
        images = glob("training_images/" + category + "/" + subcategory + "/*.png", recursive=True)

    # For ordering purposes - reindex the categories
    return DataFrame([{"category": category, "subcategory": subcategory, "image": image} for image in images])[["category", "subcategory", "image"]]
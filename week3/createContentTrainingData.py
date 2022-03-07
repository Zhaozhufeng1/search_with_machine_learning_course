import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# Do some setup work
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('popular')

def transform_name(product_name):
    # IMPLEMENT
    return normalize_text(product_name)

def normalize_text(text: str, stem: bool = True, remove_stop_words: bool = True, lower_case: bool = True,
                   remove_digits: bool = True, remove_punctuation: bool = True) -> str:
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    text = text.replace("®", "").replace("™", "")
    if remove_digits:
        chars = [c for c in text if not c.isdigit()]
        text = "".join(chars)
    if lower_case:
        results = word_tokenize(text.lower())
    else:
        results = word_tokenize(text)
    if remove_stop_words:
        results = [w for w in results if w not in stop_words]
    if remove_punctuation:
        punctuation = ".!?()[]{}\\`/-.':`,<>.".split()
        punctuation.append("''")
        punctuation.append("-")
        punctuation = set(punctuation)
        results = [w for w in results if w not in punctuation]
    if stem:
        results = [stemmer.stem(t) for t in results]
    return " ".join(results)

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      output.write("__label__%s %s\n" % (cat, transform_name(name)))

#Updating min_products
#import pandas as pd
#data = pd.read_csv(output_file, delimiter="\t", names=["categories", "products"])
#data = data.groupby(["categories"]).filter(lambda x: len(x) >= min_products)
#data.to_csv(output_file, sep="\t",header=False, index=False) 
#
# A simple endpoint that can receive documents from an external source, mark them up and return them.  This can be useful
# for hooking in callback functions during indexing to do smarter things like classification
#
from flask import (
    Blueprint, request, abort, current_app, jsonify
)
import fasttext
import json

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

threshold=0.9

bp = Blueprint('documents', __name__, url_prefix='/documents')

# Take in a JSON document and return a JSON document
@bp.route('/annotate', methods=['POST'])
def annotate():
    if request.mimetype == 'application/json':
        the_doc = request.get_json()
        response = {}
        cat_model = current_app.config.get("cat_model", None) # see if we have a category model
        syns_model = current_app.config.get("syns_model", None) # see if we have a synonyms/analogies model
        # We have a map of fields to annotate.  Do POS, NER on each of them
        sku = the_doc["sku"]
        for item in the_doc:
            the_text = the_doc[item]
            if the_text is not None and the_text.find("%{") == -1:
                if item == "name":
                    if syns_model is not None:
                        #print("IMPLEMENT ME: call nearest_neighbors on your syn model and return it as `name_synonyms`")
                        #model = fasttext.load_model("/workspace/datasets/fasttext/phone_model.bin")
                        #similar_words = model.get_nearest_neighbors(normalize_text(the_text))
                        nns = syns_model.get_nearest_neighbors(transform_name(the_text))
                        response["name_synonyms"] = [nn[1] for nn in nns if nn[0] > threshold]                       
        return jsonify(response)
    abort(415)

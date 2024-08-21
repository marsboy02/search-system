from flask import Flask, request, jsonify

from embedding import embed_sentence
from milvus import create_milvus_collection_and_index, insert_embedding, vector_search

# flask init
app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False

# milvus init
milvus = create_milvus_collection_and_index()


@app.route('/')
def healthcheck():
    return 'Healthy!'


@app.route('/sentence/<sentence>', methods=['GET'])
def search_sentence(sentence):
    embedding = embed_sentence(sentence)
    results = vector_search(milvus, embedding)
    return jsonify({"result": results, "statusCode": 200})


@app.route('/sentence', methods=['POST'])
def post_sentence():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    sentence = data.get('sentence')

    if not sentence:
        return jsonify({"error": "No 'name' field provided in JSON data"}), 400
    embedding = embed_sentence(sentence)
    insert_embedding(milvus, sentence, embedding)

    return jsonify({"message": "Data inserted successfully!", "statusCode": 200})


if __name__ == '__main__':
    app.run(port=8080, debug=True)

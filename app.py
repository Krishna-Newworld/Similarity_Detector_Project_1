from flask import Flask, request, jsonify
from flask_cors import CORS
from models.predcat import predict_with_all_models
from weightedavg import compute_weighted_score

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        s1 = data.get('sentence1')
        s2 = data.get('sentence2')

        if not s1 or not s2:
            return jsonify({'error': 'Both sentence1 and sentence2 are required.'}), 400

        predictions = predict_with_all_models(s1, s2)

        print(predictions)

        #Tokenize sentences into words (lowercased for fairness)
        #else if words are in upper case, they won't be catched even if common
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        #calculate common and unique words
        common_words = list(words1 & words2)
        unique_to_s1 = list(words1 - words2)
        unique_to_s2 = list(words2 - words1)

        final_score = compute_weighted_score(predictions)

        return jsonify({
            'sentence1': s1,
            'sentence2': s2,
            'predictions': predictions,
            'analysis' : {
            'common_words': common_words,
            'unique_to_s1': unique_to_s1,
            'unique_to_s2': unique_to_s2
            },
            'final_score' : final_score
        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

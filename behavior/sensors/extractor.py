import spacy
from transformers import pipeline
import numpy as np

# Load English Spacy for LSM
nlp = spacy.load("en_core_web_sm")

class SolidFeatureExtractor:
    def __init__(self):
        # 1. VAD / Emotion Extraction
        # We switch to a robust public model for emotion detection
        # This model covers 28 emotions which we map to VAD
        self.emotion_pipe = pipeline(
            "text-classification", 
            model="SamLowe/roberta-base-go_emotions", 
            top_k=None
        )
        
        # 2. OCEAN Extraction (Personality)
        # Using a reliable public model for Big Five
        self.ocean_pipe = pipeline(
            "text-classification", 
            model="Minej/bert-base-personality",
            top_k=None
        )

        self.lsm_tags = ['PRON', 'DET', 'ADP', 'CONJ', 'CCONJ', 'SCONJ', 'ADV', 'AUX', 'PART']

    def get_vad_from_emotions(self, emotion_results):
        """
        Simple mapping of GoEmotions to VAD scale.
        Valence: Positive vs Negative emotions
        Arousal: Intensity of the emotion
        Dominance: Assertive vs Passive
        """
        v, a, d = 0.5, 0.5, 0.5
        scores = {res['label']: res['score'] for res in emotion_results}
        
        # Simple heuristic mapping for MVP
        pos = scores.get('joy', 0) + scores.get('love', 0) + scores.get('approval', 0)
        neg = scores.get('anger', 0) + scores.get('fear', 0) + scores.get('sadness', 0)
        
        v = 0.5 + (pos - neg) * 0.5
        a = 0.5 + (scores.get('anger', 0) + scores.get('fear', 0) + scores.get('excitement', 0)) * 0.5
        d = 0.5 + (scores.get('anger', 0) + scores.get('approval', 0) - scores.get('fear', 0)) * 0.5
        
        return np.clip([v, a, d], 0, 1)

    def get_lsm_vector(self, text):
        doc = nlp(text.lower())
        counts = {tag: 0 for tag in self.lsm_tags}
        for token in doc:
            if token.pos_ in counts:
                counts[token.pos_] += 1
        total = len(doc) if len(doc) > 0 else 1
        return np.array([counts[tag] / total for tag in self.lsm_tags])

    def get_scores(self, text):
        # Emotion to VAD
        emotion_res = self.emotion_pipe(text)[0]
        v, a, d = self.get_vad_from_emotions(emotion_res)
        
        # OCEAN
        ocean_res = self.ocean_pipe(text)[0]
        raw_map = {res['label']: res['score'] for res in ocean_res}

        # Mapping strict pour Minej/bert-base-personality
        
        ocean_vector = [
            raw_map.get('LABEL_4', 0.5),  # Openness
            raw_map.get('LABEL_1', 0.5),  # Conscientiousness
            raw_map.get('LABEL_2', 0.5),  # Extraversion
            raw_map.get('LABEL_0', 0.5),  # Agreeableness
            raw_map.get('LABEL_3', 0.5)   # Neuroticism
        ]
        return {
            "v": float(v),
            "a": float(a),
            "d": float(d),
            "ocean": ocean_vector,
            "lsm_vec": self.get_lsm_vector(text).tolist()
        }

if __name__ == "__main__":
    extractor = SolidFeatureExtractor()
    test_text = "I’ve been sitting here for three hours, \
    staring at the digital blueprint of the new cultural center, \
    and for the first time in a decade, the lines don’t seem to speak back to me. \
    It’s a strange sensation, \
    this sudden silence in a mind that usually races at a hundred miles per hour. \
    Usually, I can see the flow of people through a corridor before the walls are even rendered. \
    I can feel the weight of the cantilevered roof in my own shoulders. \
    But tonight, there’s just a void. \
    Is it burnout, or is it something deeper? \
    A shift in the tectonic plates of my own ambition?"
    print(extractor.get_scores(test_text))
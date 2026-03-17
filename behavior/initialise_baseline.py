import json
import numpy as np
from behavior.sensors.extractor import SolidFeatureExtractor

def initialize_user_baseline(user_id, long_text, extractor=None, save_to_file: bool = True, save_path: str = "data/truth_repo.json"):
    if extractor is None:
        extractor = SolidFeatureExtractor()
    
    # 1. Split the text into blocks of ~250 words for stability
    words = long_text.split()
    chunks = [" ".join(words[i:i + 250]) for i in range(0, len(words), 250)]
    
    all_vectors = []
    print(f"Analyzing {len(chunks)} segments to establish truth referential...")

    for i, chunk in enumerate(chunks):
        scores = extractor.get_scores(chunk)
        # Vector in 8 dimensions : V, A, D + 5  OCEAN traits
        vector = [scores['v'], scores['a'], scores['d']] + scores['ocean']
        all_vectors.append(vector)
        print(f"Segment {i+1}/{len(chunks)} processed.")

    # 2. Calculation of reference statistics
    matrix = np.array(all_vectors)
    baseline_mean = np.mean(matrix, axis=0).tolist()
    baseline_std = np.std(matrix, axis=0).tolist()

    # 3. Build Truth Repository payload
    truth_data = {
        "user_id": user_id,
        "baseline": {
            "mean": baseline_mean,
            "std": baseline_std,
            "sample_size": len(chunks)
        },
        "lsm_reference": scores['lsm_vec']  # Last LSM profile as the base
    }

    if save_to_file:
        with open(save_path, 'w') as f:
            json.dump(truth_data, f, indent=4)
        print("\n--- BASELINE ESTABLISHED ---")
        print(f"Mean Vector: {baseline_mean}")
        print(f"Saved to {save_path}")

    return truth_data

# TEST
long_text = """I’ve been sitting here for three hours, staring at the digital blueprint of the new cultural center, and for the first time in a decade, the lines don’t seem to speak back to me. It’s a strange sensation, this sudden silence in a mind that usually races at a hundred miles per hour. Usually, I can see the flow of people through a corridor before the walls are even rendered. I can feel the weight of the cantilevered roof in my own shoulders. But tonight, there’s just a void. Is it burnout, or is it something deeper? A shift in the tectonic plates of my own ambition?

When I started this journey, I was obsessed with the minute details. I remember my first mentor telling me that architecture is the marriage of math and poetry, but I always leaned more toward the math. I liked the certainty of it. If the load-bearing calculations were correct, the building stood. It was a binary world. But as the years have passed, I’ve found myself increasingly drawn to the poetry—the intangible way light hits a concrete surface at 4:00 PM in mid-October, or the way a space can force a stranger to exhale. This shift has made me more creative, certainly, but it has also made me more anxious. The math never lied to me, but the poetry is subjective, fickle, and prone to failure.

I think about my team often. They look to me for a level of extroverted leadership that I find exhausting to maintain. I can play the part of the visionary director during a client pitch—I can project confidence, wave my hands to illustrate grand volumes, and speak in the charismatic cadences that people expect from a 'starchitect.' But as soon as the door closes, I feel like a collapsed star. I need the solitude. I need the quiet of my studio where the only sound is the hum of my computer and the occasional scratch of a pencil. It’s a paradox, isn't it? To build spaces for thousands of people while wishing to remain unseen by almost all of them.

Last week, during the public hearing, someone asked me why I chose such an unconventional material for the facade. I launched into a twenty-minute technical explanation about thermal mass and recycled polymer composites, but what I really wanted to say was that it felt right. It felt like the future. I’ve always been high on openness to new experiences—sometimes to a fault. I’d rather fail brilliantly with a new experimental technique than succeed with a boring, tried-and-true method. My business partner hates this about me. He’s the 'Conscientiousness' to my 'Openness.' He’s the one who checks the insurance contracts and ensures we don't go 400% over budget. Without him, I’d probably be a brilliant, homeless dreamer.

I wonder if I’m becoming more neurotic as I get older. I find myself checking the structural integrity reports three, four, five times. Not because I don't trust the engineers, but because I’ve started to visualize the 'what ifs.' What if the soil samples were flawed? What if the seismic dampers have a latent manufacturing defect? This anxiety is a double-edged sword. It keeps the buildings safe, but it keeps me awake at 3:00 AM, pacing the hardwood floors of my apartment, questioning every decision I’ve made since university.

Sometimes, I miss the simplicity of being a student. Back then, the world was a canvas. We would sit in dive bars until dawn, arguing about Le Corbusier and Zaha Hadid as if the fate of the universe depended on it. We were so agreeable back then—or perhaps we were just equally naive. Now, professional life has sharpened my edges. I’ve had to become more disagreeable to protect my designs from being watered down by developers who only care about the cost per square foot. Every building is a battleground, and I am a tired soldier.

But then, I look at the sketches again. I see that one corner where the light will pool in the morning. I imagine a child walking through that lobby twenty years from now and feeling, for just a second, that the world is a beautiful and intentional place. And the exhaustion lifts, just for a moment. The neuroticism fades into the background, the introverted fatigue is forgotten, and the math and poetry finally start to rhyme again.

Maybe the dilemma isn't about whether I'm the right person for this task. Maybe the dilemma is simply that I care too much. I see the 'Agreeableness' in my willingness to listen to the community's concerns, even when they conflict with my aesthetic vision. I see the 'Neuroticism' in my fear of failure. I see the 'Openness' in the glass ceilings I want to build. It’s all there, woven into the fabric of my personality, and therefore, woven into the buildings I leave behind.

I should probably sleep. The presentation is in six hours. My partner will be there, perfectly dressed, perfectly prepared, leaning on his conscientiousness like a shield. I will be there, slightly disheveled, leaning on my vision and hoping the client can see the poetry through the spreadsheets. That is the dance we do. That is the architecture of a life spent building things that are meant to outlast us.

If this center gets built, it won't just be a structure of steel and glass. It will be a map of my own internal struggles. Every beam a choice, every window a doubt. I hope they like the facade. I hope the math holds. But mostly, I hope the light hits the floor exactly the way I imagine it will. Because if it does, all of this—the anxiety, the isolation, the endless questioning—will have been the price worth paying."""

if __name__ == "__main__":
    initialize_user_baseline("user_01", long_text)
import whisper
import ffmpeg
from tqdm import tqdm
import math
import re
from collections import Counter
import string
import numpy as np
import pandas as pd

LIMIT = 1000


def grab_keywords(sentence, wlen):
    words = sentence.split(" ")
    keywords = []
    for i in words:
        if len(i) >= wlen:
            keywords.append(i)
    a = " ".join(keywords)
    return a

def cosinesimularity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x]**2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

def get_cosinesimularity(text1, text2):
    vector1 = text_to_vector(text1.lower().translate(str.maketrans('', '', string.punctuation)))
    vector2 = text_to_vector(text2.lower().translate(str.maketrans('', '', string.punctuation)))

    cosine = cosinesimularity(vector1, vector2)

    return cosine



def load_downsample(filename):
    #downsampling the file
    samps = [8000,10000,14000]
    resampled = []
    resampled.append(whisper.load_audio(filename)) #baseline
    for i in samps:
        resampled.append(whisper.load_audio(filename, sr=i))
    
    #classifying

    resampled_preds = []
    for i in resampled:
        result = model.transcribe(i, fp16=False)
        resampled_preds.append(result["text"])
    

    return resampled_preds


if __name__ == "__main__":
    #loading in the dataset csv
    #data keys are: filename, text, up_votes, down_votes, age, gender, accent and duration
    data = pd.read_csv("cv-valid-test.csv")
    print("[!] loading model")
    model = whisper.load_model("base")
    print("[!] model loaded")
    print("[!] complete")

    acc_scores = []
    print("[*] starting downsampling experiment")
    try:
        for index, samp in tqdm(enumerate(data.iterrows())):
            g_truth = text = samp[1][1]
            file_path = samp[1][0]
            preds = load_downsample(file_path)
            scores = []
            for i in preds:
                scores.append(get_cosinesimularity(grab_keywords(i, 5),grab_keywords(g_truth, 5)))
            
            if index == LIMIT:
                break
            
            acc_scores.append(scores)
    except KeyboardInterrupt:
        print("[!] Experiment terminated samples tested {}".format(len(acc_scores)))

    print("[*] experiment complete")

    print("[*] Aggregating scores")
    median = [[],[],[],[]]
    for i in acc_scores:
        for index, s in enumerate(i):
            median[index].append(s)


    print("[*] average scores")
    for i in median:
        print(np.mean(i))

    print("[*] median scores")
    for i in median:
        print(np.median(i))
    

    print("[!] complete")
    exit()


""" 
Example output:

[!] loading model
[!] model loaded
[!] complete
[*] starting downsampling experiment
1000it [5:28:34, 19.71s/it]
[*] experiment complete
[*] Aggregating scores
[*] average scores
0.8081876201666295
0.18184186406087327
0.5901372462664584
0.7939659710658
[*] median scores
0.9128709291752769
0.0
0.6708203932499369
0.8944271909999159
[!] complete  
"""
import pandas as pd
import os
from tqdm import tqdm

headings = [
    "fname",
    "emotion",
    "intensivity",
    "statement",
    "repetition",
    "actor",
    "sex"
]

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "suprised"
}

intesivity = {
    "01": "normal",
    "02": "strong"
}

statement = {
    "01": "kids",
    "02": "dogs"
}

sex = {
    0: "female",
    1: "male"
}


def makeSeriesFromName(filename: str) -> list:

    filename = filename[filename.find('/') + 1:]
    filename = filename[: filename.find('.')]
    name = filename
    nums = name.split(sep='-')
    emotion = emotions.get(nums[2])
    intens = intesivity.get(nums[3])
    sentence = statement.get(nums[4])
    repetition = int(nums[5])
    actor = int(nums[6])
    c3 = sex.get((int(nums[6]) % 2))

    record = [
        name,
        emotion,
        intens,
        sentence,
        repetition,
        actor,
        c3
    ]

    return record


dir = os.listdir('RAVDESS/')
rows = []

for audiofile in tqdm(dir):

    rows.append(makeSeriesFromName(audiofile))


df = pd.DataFrame(rows, columns=headings)
df.to_csv("RAVDESS_db.csv")




import pandas as pd
import os
from tqdm import tqdm

headings: list = [
    "fname",
    "emotion",
    "intensivity",
    "statement",
    "repetition",
    "actor",
    "sex"
]

emotions: dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "suprised"
}

intesivity: dict = {
    "01": "normal",
    "02": "strong"
}

statement: dict = {
    "01": "kids",
    "02": "dogs"
}

sex: dict = {
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


inputDir: str = 'RAVDESS'
outputCSV: str = 'RAVDESS_db.csv'

dir = os.listdir(inputDir)
rows = []

for audiofile in tqdm(dir):
    rows.append(makeSeriesFromName(audiofile))

df = pd.DataFrame(rows, columns=headings)
df.to_csv(outputCSV, index=False)

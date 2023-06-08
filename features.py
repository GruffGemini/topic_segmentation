from string import punctuation
import numpy as np
import pandas as pd
from typing import List, Union

SMOOTHING_WINDOW = 3
CUE_PHRASES = {'ok': 1, 'okay': 1, 'kay': 1, "'kay": 1, 'so': 0.82, 'and': 0.32, 'yeah': 0.31, 'well': 0.18,
               'right': 0.12, 'but': 0.11, 'alright': 0.08, 'now': 0.06, 'anyway': 0.04}
JUNK_WORDS = ['uh', 'um', 'mm']
FEATURE_WEIGHTS = [0.1, 0.5, 0.1, 0.1, 0.2]


def get_silence_features(df: pd.DataFrame, start_col_name: str, end_col_name: str) -> List[float]:
    starts = df[start_col_name][1:].values
    ends = df[end_col_name].values
    starts = np.append(starts, 0)

    meeting_features_calc = starts - ends
    meeting_features_calc = [0.0] + [round(i, 2) for i in meeting_features_calc]
    meeting_features_calc.pop()
    meeting_features_calc = smooth_silence_features(meeting_features_calc)
    meeting_features_calc = smooth_features(meeting_features_calc)

    features = normalize(meeting_features_calc)

    return features


def get_speaker_change_features(df: pd.DataFrame, speaker_col_name: str) -> List[float]:
    meeting_features = []

    speakers = df[speaker_col_name].values
    current_speaker_duration = 0
    current_speaker = None
    for i, speaker in enumerate(speakers):
        if speaker != current_speaker:
            meeting_features.append(current_speaker_duration)
            current_speaker_duration = 1
            current_speaker = speaker
        else:
            meeting_features.append(0)
            current_speaker_duration += 1

    meeting_features = smooth_features(meeting_features)
    features = normalize(meeting_features)

    return features


def get_overlap_features(df: pd.DataFrame, speaker_col_name: str, start_col_name: str,
                         end_col_name: str) -> List[float]:
    meeting_features = [0.0]

    speakers = df[speaker_col_name].values
    starts = df[start_col_name].values
    ends = df[end_col_name].values

    i = 1
    while i < len(df):
        if speakers[i] != speakers[i - 1] and starts[i] < ends[i - 1]:
            meeting_features.append(ends[i] - starts[i])
        else:
            meeting_features.append(0.0)
        i += 1

    meeting_features = smooth_features(meeting_features)
    features = [-i for i in normalize(meeting_features)]

    return features


def get_cue_phrases_features(df: pd.DataFrame, caption_col_name: str) -> List[float]:
    meeting_features = [0.0]
    texts = df[caption_col_name].values

    for text in texts:
        tokens = text.lower().split()
        for token in tokens:
            token = token.strip(punctuation)
            if token not in JUNK_WORDS:
                if token in CUE_PHRASES:
                    meeting_features.append(CUE_PHRASES[token])
                else:
                    meeting_features.append(0.0)
                break
        else:
            meeting_features.append(0.0)

    features = smooth_features(meeting_features)

    return features


def smooth_features(features: List[Union[int, float]]) -> List[float]:
    new_features = [0.0] * len(features)
    i = 0
    while i < len(features):
        new_features[i] = max(float(features[i]), new_features[i])
        if features[i] > 0:
            intensity_fraction = features[i] / (SMOOTHING_WINDOW + 1)
            for j in range(1, SMOOTHING_WINDOW + 1):
                if i - j >= 0:
                    new_features[i - j] = max(new_features[i - j], intensity_fraction * (SMOOTHING_WINDOW - j + 1))
                if i + j < len(features):
                    new_features[i + j] = max(new_features[i + j], intensity_fraction * (SMOOTHING_WINDOW - j + 1))
        i += 1

    return new_features


def smooth_silence_features(features: List[float]) -> List[float]:
    new_features = []
    i = 0
    while i < len(features):
        if features[i] < 0:
            j = i + 1
            while j < len(features) and features[j] < 0:
                new_features.append(0.0)
                j += 1
            i = j - 1
            if i + 1 >= len(features):
                new_features.append(0)
            else:
                new_features.extend([0.0, max(0.0, round(features[i + 1] + features[i], 2))])
            i += 1
        else:
            new_features.append(features[i])
        i += 1

    return new_features


def normalize(values: List[float]) -> List[float]:
    try:
        return [(value - min(values)) / (max(values) - min(values)) for value in values]
    except ZeroDivisionError:
        return [0.5] * len(values)

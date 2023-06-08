from pandas import DataFrame
from sys import stderr
from typing import Union

START_COL_NAME = "starttime"
END_COL_NAME = "endtime"
CAPTION_COL_NAME = "text"
SPEAKER_COL_NAME = 'speaker'

RAW_START_COL_NAME = 'starttime'
RAW_END_COL_NAME = 'endtime'


def preprocessing(df, caption_col_name):
    fillers = ["um", "uh", "oh", "hmm", "you know", "like"]
    fillers += list(
        map(lambda filler: filler + " ", fillers)
    )  # filler inside caption with other words
    fillers = list(
        map(lambda filler: "(?i)" + filler, fillers)
    )  # make it case-insensitive
    df[caption_col_name].replace(fillers, [""] * len(fillers), regex=True, inplace=True)

    captions_with_multiple_sentences = len(df.loc[df[caption_col_name].isin(["."])])
    if captions_with_multiple_sentences > 0:
        print(
            f"WARNING: Found {captions_with_multiple_sentences} captions with multiple sentences; "
            f"sentence embeddings may be inaccurate.",
            file=stderr,
        )

    df = df[df[caption_col_name].str.len() > 20]
    df.reset_index(inplace=True)

    return df


def create_dataframe(text_data: Union[dict, list]) -> DataFrame:
    caption_data = []

    if isinstance(text_data, dict):
        message_list = text_data['message_list']
    elif isinstance(text_data, list):
        message_list = text_data
    else:
        raise Exception('Unknown input format: has to be dict or list')

    current_sentence = []
    start_time = None
    for message in message_list:
        current_sentence.append(message['text'].strip())
        if not start_time:
            start_time = float(message[RAW_START_COL_NAME])
        if current_sentence[-1][-1] not in '.?!':
            continue

        sentence = ' '.join(current_sentence)
        if ',' in sentence:
            sentence = f'"{sentence}"'
        caption_data.append([sentence, start_time, float(message[RAW_END_COL_NAME]), message['speaker']])
        current_sentence = []
        start_time = None

        if current_sentence:
            sentence = ' '.join(current_sentence)
            if ',' in sentence:
                sentence = f'"{sentence}"'
            caption_data.append([sentence, start_time, message['end_time'], message['speaker']])

    return DataFrame(data=caption_data, columns=[CAPTION_COL_NAME, START_COL_NAME, END_COL_NAME, SPEAKER_COL_NAME])

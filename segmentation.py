import json
import torch
from typing import NamedTuple, Optional, Literal, Dict
from tqdm.auto import tqdm

from dataset import *
from features import *
from models import MBertModel, SBertModel, CommonModelName


class TextTilingHyperparameters(NamedTuple):
    SENTENCE_COMPARISON_WINDOW: int = 15
    SMOOTHING_PASSES: int = 2
    SMOOTHING_WINDOW: int = 1


class TopicSegmentationConfig(NamedTuple):
    TEXT_TILING: Optional[TextTilingHyperparameters] = TextTilingHyperparameters()
    MAX_SEGMENTS_CAP: bool = True
    MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH: int = 60
    MIN_SEGMENT_LENGTH = 10


def segment_text(text_data: Union[dict, str, list], model: Literal[CommonModelName.SBERT, CommonModelName.MBERT],
                 human_report: str, threshold: float = 0.6, use_features=True) -> List[Dict[str, float]]:
    """
    The main function that performs text segmentation. Returns a list of timestamps for every chapter.
    Each timestamp is a dict with fields 'start_time' and 'end_time'. The times themselves are float numbers.

        - text_data
        Can be one of the following:
            - transcript json as a dict. The field "message_list" needs to be present in this JSON
            - a list of message lists (a list of dicts). Those dicts require fields "start_time", "end_time", "text"
            - a string that can be converted to any of the above two types

        - model
        A literal that determines what kind of embeddings will be used for segmentation. SBERT and
        bert-multilingual-cased are supported

        - threshold
        A threshold that determines how significant should be the change in dialogue semantics to regard it as a topic
        change. Should be a real number between 0 and 1. 0 will create a lot of small chapters. 1 will cause the entire
        dialogue to be a single chapter. Default value is 0.5

        - use_features
        If set to true, linguistic features will be used for segmentation in addition to BERT embeddings. If set to
        false, only BERT embeddings will be used
    """

    try:
        input_data = json.loads(text_data)
    except TypeError:
        input_data = text_data

    data = create_dataframe(input_data)

    if model == CommonModelName.MBERT:
        model = MBertModel(threshold=threshold)
    elif model == CommonModelName.SBERT:
        model = SBertModel(threshold=threshold)
    else:
        raise Exception('Unknown model name!')

    bert_segments, bert_features = topic_segmentation_bert(df=data,
                                                           caption_col_name=CAPTION_COL_NAME,
                                                           start_col_name=START_COL_NAME,
                                                           end_col_name=END_COL_NAME,
                                                           model=model)

    if not use_features:
        segments = bert_segments
    else:
        side_window = (TextTilingHyperparameters().SMOOTHING_WINDOW +
                       TextTilingHyperparameters().SENTENCE_COMPARISON_WINDOW)
        bert_features = [min(bert_features)] * side_window + bert_features + [min(bert_features)] * side_window
        bert_features = normalize(bert_features)

        cue_phrases_features = get_cue_phrases_features(df=data, caption_col_name=CAPTION_COL_NAME)
        overlap_features = get_overlap_features(df=data,
                                                start_col_name=START_COL_NAME,
                                                end_col_name=END_COL_NAME,
                                                speaker_col_name=SPEAKER_COL_NAME)
        speaker_change_features = get_speaker_change_features(df=data, speaker_col_name=SPEAKER_COL_NAME)
        silence_features = get_silence_features(df=data, start_col_name=START_COL_NAME, end_col_name=END_COL_NAME)

        bert_weights = [i * FEATURE_WEIGHTS[0] for i in bert_features]
        silence_weights = [i * FEATURE_WEIGHTS[1] for i in silence_features]
        overlap_weights = [i * FEATURE_WEIGHTS[2] for i in overlap_features]
        speaker_change_weights = [i * FEATURE_WEIGHTS[3] for i in speaker_change_features]
        cue_phrases_weights = [i * FEATURE_WEIGHTS[4] for i in cue_phrases_features]

        zipped_features = zip(bert_weights, silence_weights, overlap_weights, speaker_change_weights,
                              cue_phrases_weights)
        total_features = normalize([sum(item) for item in zipped_features])
        segments = features_to_prediction(total_features, threshold=threshold)

    segments = sorted(segments)

    if human_report:
        make_human_report(input_df=data, output_filename=human_report, predictions=segments)

    report = []
    for i, _ in enumerate(segments):
        segment_start_time = data.iloc[segments[i]].starttime
        segment_end_time = data.iloc[segments[i + 1] - 1].endtime if i != len(segments) - 1 else data.iloc[-1].endtime

        report.append({'start_time': float(segment_start_time), 'end_time': float(segment_end_time)})

    return report


def split_list(a, n):
    """
    a utility function that is used to split a given list into smaller lists of a specified size.
    :list a:
    :an integer n:
    :return a generator that yields n smaller lists, each containing a portion of the elements from the input list

    Here is an example of how the split_list function could be used:

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = 3

    for small_list in split_list(a, n):
        print(small_list)

    # Output:
    # [1, 2, 3, 4]
    # [5, 6, 7]
    # [8, 9, 10]

    """
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(min(len(a), n))
    )


def flatten_features(batches_features):
    res = []
    for batch_features in batches_features:
        res += batch_features
    return res


def get_timeseries(caption_indexes, features):
    timeseries = []
    for caption_index in caption_indexes:
        timeseries.append(features[caption_index])
    return timeseries


def topic_segmentation_bert(
        df: DataFrame,
        start_col_name: str,
        end_col_name: str,
        caption_col_name: str,
        model
):
    topic_segmentation_configs = TopicSegmentationConfig()
    textiling_hyperparameters = topic_segmentation_configs.TEXT_TILING

    # parallel inference
    batches_features = []
    for batch_sentences in tqdm(list(split_list(
            df[caption_col_name], model.PARALLEL_INFERENCE_INSTANCES
    ))):
        batches_features.append(model.get_features_from_sentence(batch_sentences))
    features = flatten_features(batches_features)

    caption_indexes = list(df.index)

    timeseries = get_timeseries(caption_indexes, features)
    block_comparison_score_timeseries = block_comparison_score(
        timeseries, k=textiling_hyperparameters.SENTENCE_COMPARISON_WINDOW
    )

    block_comparison_score_timeseries = smooth(
        block_comparison_score_timeseries,
        n=textiling_hyperparameters.SMOOTHING_PASSES,
        s=textiling_hyperparameters.SMOOTHING_WINDOW,
    )

    depth_score_timeseries = depth_score(block_comparison_score_timeseries)

    meeting_start_time = df[start_col_name].iloc[0]
    meeting_end_time = df[end_col_name].iloc[-1]
    meeting_duration = meeting_end_time - meeting_start_time
    segments = depth_score_to_topic_change_indexes(
        model,
        depth_score_timeseries,
        meeting_duration,
        topic_segmentation_configs=topic_segmentation_configs,
    )

    side_window = textiling_hyperparameters.SMOOTHING_WINDOW + textiling_hyperparameters.SENTENCE_COMPARISON_WINDOW
    segments = [i + side_window for i in segments]

    if len(segments) == 0:
        segments = np.empty(0, int)
    segments = np.append(segments, 0)

    return segments, depth_score_timeseries


# FUNCTIONS THAT ARE USED IN `TOPIC_SEGMENTATION_BERT`

def block_comparison_score(timeseries, k):
    """
    comparison score for a gap (i)
    cfr. docstring of block_comparison_score
    """
    res = []
    for i in range(k, len(timeseries) - k):
        first_window_features = compute_window(timeseries, i - k, i + 1)
        second_window_features = compute_window(timeseries, i + 1, i + k + 2)
        res.append(
            sentences_similarity(first_window_features, second_window_features)
        )

    return res


def compute_window(timeseries, start_index, end_index):
    """given start and end index of embedding, compute pooled window value
    [window_size, 768] -> [1, 768]
    """
    stack = torch.stack([features for features in timeseries[start_index:end_index]])
    stack = stack.unsqueeze(
        0
    )  # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
    stack_size = end_index - start_index
    pooling = torch.nn.MaxPool2d((stack_size - 1, 1))  # CHECK FOR SENSE?!
    return pooling(stack).squeeze(0)


def sentences_similarity(first_sentence_features, second_sentence_features) -> float:
    """
    Given two sentences embedding features compute cosine similarity
    """
    similarity_metric = torch.nn.CosineSimilarity()
    return float(similarity_metric(first_sentence_features, second_sentence_features))


def smooth(timeseries, n, s):
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                         max(0, index - s): min(len(timeseries) - 1, index + s)
                         ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def depth_score(timeseries):
    """
    The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
    given token-sequence gap and is based on the distance from the peaks on both sides of the valley to that valley.
    returns depth_scores
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
                right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return depth_scores


def depth_score_to_topic_change_indexes(
        model,
        depth_score_timeseries,
        meeting_duration,
        topic_segmentation_configs
):
    """
    capped add a max segment limit so there are not too many segments, used for UI improvements on the Workplace
    TeamWork product
    """

    if not depth_score_timeseries:
        return []

    capped = topic_segmentation_configs.MAX_SEGMENTS_CAP
    average_segment_length = (
        topic_segmentation_configs.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH
    )
    threshold = model.threshold * max(depth_score_timeseries)

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)

    if not local_maxima:
        return []

    if capped:  # capped is segmentation used for UI
        # sort based on maxima for pruning
        local_maxima, local_maxima_indices = arsort2(local_maxima, local_maxima_indices)

        # local maxima are sorted by depth_score value and we take only the first K
        # where the K+1th local maxima is lower then the threshold
        for thres in range(len(local_maxima)):
            if local_maxima[thres] <= threshold:
                break

        max_segments = int(meeting_duration / average_segment_length)
        slice_length = min(max_segments, thres)

        local_maxima_indices = local_maxima_indices[:slice_length]
        local_maxima = local_maxima[:slice_length]

        # after pruning, sort again based on indices for chronological ordering
        local_maxima_indices, _ = arsort2(local_maxima_indices, local_maxima)

    else:  # this is the vanilla TextTiling used for Pk optimization
        filtered_local_maxima_indices = []
        filtered_local_maxima = []

        for i, m in enumerate(local_maxima):
            if m > threshold:
                filtered_local_maxima.append(m)
                filtered_local_maxima_indices.append(i)

        local_maxima_indices = filtered_local_maxima_indices

    return local_maxima_indices


def get_local_maxima(array):
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def arsort2(array1, array2):
    x = np.array(array1)
    y = np.array(array2)

    sorted_idx = x.argsort()[::-1]
    return x[sorted_idx], y[sorted_idx]


def check_topic_break(topic_break: int, current_breaks: List[int]) -> bool:
    min_segment_length = TopicSegmentationConfig().MIN_SEGMENT_LENGTH
    for i in current_breaks:
        if abs(i - topic_break) < min_segment_length:
            return False
    return True


def features_to_prediction(features: List[float], threshold) -> List[str]:
    sorted_features = sorted(list(enumerate(features)), key=lambda x: x[1], reverse=True)
    topic_breaks = [0]
    for i, feature in sorted_features:
        if (feature > threshold and
                check_topic_break(topic_break=i, current_breaks=topic_breaks)):
            topic_breaks.append(i)

    return topic_breaks


def make_human_report(input_df: DataFrame, predictions: List[int], output_filename: str):
    with open(output_filename, 'w') as f:
        prev_speaker = None
        predictions_chapter = 0
        for index, row in input_df.iterrows():
            if index in predictions:
                if predictions_chapter != 0:
                    f.write('\n\n')
                predictions_chapter += 1
                f.write(f'Chapter {predictions_chapter}\n')
                prev_speaker = None

            if row['speaker'] != prev_speaker:
                f.write(f'\n{row["speaker"]}: ')
                prev_speaker = row['speaker']
            f.write(f'{row["text"]} ')

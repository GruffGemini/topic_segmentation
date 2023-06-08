# topic_segmentation

A tool for performing unsupervised topic segmentation of work meetings. Based on multilingual BERT and SBERT
and https://github.com/gdamaskinos/unsupervised_topic_segmentation

## Sample Usage:

```
python main.py sample_input/ES2002c.json report.txt -t 0.6 -r human_report.txt -m sbert -f
```

## Input parameters:

- input file. Has to be either a transcript JSON with `message_list` field or a list of message JSONs with
  fields `starttime`, `endtime`, `text`.
- output_file. The result will be written as a list of timestamp dicts, each of which denotes the start and the end
  times of each chapter
- -m (model). Available options are `sbert` or `mbert` (bert-multilingual-cased will be used)
- [Optional] -t (threshold). A float number from 0 to 1. Determines how significant should be the change in dialogue
  semantics to regard it as a topic change. Should be a real number between 0 and 1. 0 will create a lot of small
  chapters. 1 will cause the entire dialogue to be a single chapter. Default value is 0.5 for sbert and 0.4 for mbert.
- [Optional] -p (preprocessing). Pass this argument to perform preprocessing on the input data. Preprocessing removes
  fillers and very short sentences from the dialogue. Can help if the data is noisy but can also drop potentially useful
  information. Disabled by default.
- [Optional] -r (human_report). Specify a path to a human-readable segmentation report.
- [Optional] -f (features). Use linguistic features for segmentation. Is is recommended to enable this option for better
  results. It is not recommended enabling preprocessing while using linguistic features.

## Internal usage

The entry point function for segmentation task is `segmentation.segment_text`
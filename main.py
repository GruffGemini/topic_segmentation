import json
from argparse import ArgumentParser

from models import CommonModelName
from segmentation import segment_text

parser = ArgumentParser()

parser.add_argument('input_file',
                    help='Path to the input file that contains either a transcript json or a message list.')
parser.add_argument('output_file',
                    help='Path to the output file with chapter markup.')
parser.add_argument('-m', '--model', required=True, choices=['mbert', 'MBERT', 'bert', 'BERT', 'xlm', 'XLM'],
                    help='Model to create sentence embeddings with. Available options are XLM or MBERT '
                         '(bert-multilingual-cased).')
parser.add_argument('-t', '--threshold', required=False, default=0.5, type=float,
                    help='[Optional] A threshold that determines how significant should be the change in dialogue '
                         'semantics to regard it as a topic change. Should be a real number between 0 and 1. '
                         '0 will create a lot of small chapters. 1 will cause the entire dialogue to be '
                         'a single chapter. Default value is 0.5.')
parser.add_argument('-p', '--preprocessing', required=False, action='store_true',
                    help='[Optional] Pass this argument to perform preprocessing on the input data. Preprocessing '
                         'removes fillers and very short sentences from the dialogue. Can help if the data is noisy '
                         'but can also drop potentially useful information. Disabled by default.')

args = parser.parse_args()

if args.threshold < 0 or args.threshold > 1:
    print('ERROR: threshold must be between 0 and 1')
    exit(1)

common_model_name = CommonModelName.MBERT if args.model.lower() in ['bert', 'mbert'] else CommonModelName.XLM

with open(args.input_file) as f:
    input_data = f.read()

report = segment_text(input_data, model=common_model_name, threshold=args.threshold)

with open(args.output_file, 'w') as f:
    f.write(json.dumps(report, indent=4))

#! /usr/bin/env python
import json
import os
from argparse import ArgumentParser

from dnee.events.extract_events import extract_multi_faceted_events
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus-path", type=str, required=True)
    args = parser.parse_args()

    for example_id in tqdm(os.listdir(args.corpus_path)):
        ann_filepath = os.path.join(
            args.corpus_path, example_id, "corenlp-annotation.json"
        )
        if not os.path.exists(ann_filepath):
            continue
        with open(ann_filepath) as fp:
            ann_dict = json.load(fp)
        events_dict = extract_multi_faceted_events(ann_dict, no_sentiment=True)
        sentences = [
            [token["originalText"] for token in sentence["tokens"]]
            for sentence in ann_dict["sentences"]
        ]
        output_filepath = os.path.join(
            args.corpus_path, example_id, "dnee.json"
        )
        with open(output_filepath, "w") as fp:
            json.dump({"sentences": sentences, "events": events_dict}, fp)


if __name__ == "__main__":
    main()

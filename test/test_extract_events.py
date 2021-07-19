import json

import pytest
from dnee.events.extract_events import extract_multi_faceted_events


@pytest.mark.parametrize(
    "document_path",
    [
        "/home/wsc/public/script-event-dataset/nyt-corenlp/corpus/nyt_story_AFE19940520.0244/corenlp-annotation.json",  # noqa
        "/home/wsc/public/script-event-dataset/nyt-corenlp/corpus/nyt_story_AFE19940520.0244/corenlp-annotation.json",  # noqa
    ],
)
def test_extract_events(document_path: str):
    with open(document_path) as fp:
        dict_events = json.load(fp)

    dict_events = extract_multi_faceted_events(dict_events, no_sentiment=True)
    has_object = any(
        "arg1" in event for events in dict_events.values() for event in events
    )
    del dict_events

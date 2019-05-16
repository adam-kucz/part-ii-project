from collections import Counter
from pprint import pprint
from typing import Callable, Iterable, List, Optional

from funcy import map, walk_values, group_by, ilen, cached_property, isa, all
import numpy as np

from .util import pickable_option
from .predictions import Predictions, RecordWithPrediction, RecordMode


record_confidence = RecordWithPrediction.confidence
record_most_likely = RecordWithPrediction.most_likely


def random_perm(elems: List):
    perm = np.random.permutation(range(len(elems)))
    return (elems[i] for i in perm)


class Cases(Predictions):
    vocab: List[str]

    def print_basic(self, record):
        inputs = (record.identifier if self.mode == RecordMode.IDENTIFIER
                  else record.inputs)
        print("{}, {} ({}%)".format(inputs, record.label, record.confidence))

    def print_details(self, record, k=3):
        inputs = (record.identifier if self.mode == RecordMode.IDENTIFIER
                  else record.inputs)
        print("{}, {}, top {} predictions: "
              .format(inputs, record.label, k), end='')
        print(", ".join("{} ({}%)".format(val, perc)
                        for val, perc in record.predictions[:k]))

    def print_unique_by(
            self, iterable: Iterable, key=lambda r: r.identifier,
            printer: Optional[Callable[[RecordWithPrediction], None]] = None,
            n: int = 10):
        printer = printer or self.print_details
        examples = set()
        for record in iterable:
            assert isinstance(record, RecordWithPrediction), record
            if key(record) in examples:
                continue
            examples.add(key(record))
            printer(record)
            if len(examples) >= n:
                break

    @cached_property
    def most_sure(self):
        most_sure = map(lambda t: max(t[1], key=record_confidence),
                        group_by(lambda r: r.identifier, self.correct).items())
        return sorted(most_sure, key=record_confidence, reverse=True)

    @cached_property
    def most_wrong(self):
        most_wrong = map(lambda t: max(t[1], key=record_confidence),
                         group_by(lambda r: r.identifier, self.wrong).items())
        return sorted(most_wrong, key=record_confidence, reverse=True)

    @pickable_option
    def show_most_sure(self):
        print("Most sure about:")
        self.print_unique_by(self.most_sure)

    @pickable_option
    def show_most_wrong(self):
        print("Most wrong about:")
        self.print_unique_by(self.most_wrong)

    @pickable_option
    def show_most_unsure(self):
        print("Correct, but very unsure:")
        self.print_unique_by(reversed(self.most_sure))

    @pickable_option
    def show_least_wrong(self):
        print("Wrong, justifiably unsure:")
        self.print_unique_by(reversed(self.most_wrong))

    @pickable_option
    def show_correct_random(self):
        print("Correct, random:")
        assert all(isa(RecordWithPrediction), self.correct)
        self.print_unique_by(random_perm(self.correct))

    @pickable_option
    def show_wrong_random(self):
        print("Wrong, random:")
        self.print_unique_by(random_perm(self.wrong))

    def show_most_for_types(self):
        raise NotImplementedError
        for typ, records in group_by(record_most_likely, self.records):
            correct_records = (r for r in records if r.correct(self.vocab))
            grouped_by_id = group_by(lambda r: r.identifier, correct_records)
            unique_id = map(lambda t: max(t[1], key=record_confidence),
                            grouped_by_id.items())
            most_sure = sorted(unique_id, key=record_confidence, reverse=True)
            print(f"For type {typ} most sure about:")
            self.print_unique_by(most_sure)
            print()
            print(f"For type {typ} correct, but very unsure:")
            self.print_unique_by(reversed(most_sure))
            print()
            print(f"For type {typ} correct, random:")
            self.print_unique_by(random_perm(most_sure))
            print()
            wrong_records = (r for r in records if r.correct(self.vocab))
            grouped_by_id = group_by(lambda r: r.identifier, wrong_records)
            unique_id = map(lambda t: max(t[1], key=record_confidence),
                            grouped_by_id.items())
            most_wrong = sorted(unique_id, key=record_confidence, reverse=True)
            print(f"For type {typ} most wrong about:")
            self.print_unique_by(most_wrong)
            print(f"For type {typ} wrong, justifiably unsure:")
            self.print_unique_by(reversed(most_wrong))
            print()
            print(f"For type {typ} wrong, random:")
            self.print_unique_by(random_perm(most_sure))
            print()

    @pickable_option
    def predicted_types_stats(self):
        print("Numbers of predicted types:")
        pprint(walk_values(len, dict(group_by(record_most_likely,
                                              self.records))))

    @pickable_option
    def correct_predicted_types_stats(self):
        print("Numbers of predicted types in correct records:")
        pprint(walk_values(len, dict(group_by(record_most_likely,
                                              self.correct))))

    @pickable_option
    def wrong_predicted_types_stats(self):
        print("Numbers of predicted types in wrong records:")
        pprint(walk_values(lambda records:
                           Counter(r.label for r in records).most_common(),
                           dict(group_by(record_most_likely, self.wrong))))

    @pickable_option
    def show_accuracy(self):
        accuracy = len(self.correct) / len(self.records)
        real_accuracy\
            = (ilen(r for r in self.records
                    if r.label in self.vocab and r.correct(self.vocab))
               / ilen(r for r in self.records if r.label in self.vocab))
        print("Accuracy: {}%, real accuracy: {}%"
              .format(100 * accuracy, 100 * real_accuracy))

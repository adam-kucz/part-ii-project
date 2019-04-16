from pprint import pprint
import random
from typing import Callable, Iterable, List, Optional

from funcy import map, walk_values, group_by, ilen, cached_property

from .util import with_pickable_options, pickable_option
from .predictions import Predictions, Record, RecordMode


def random_from(sequence) -> Iterable:
    while True:
        yield random.choice(sequence)


@with_pickable_options
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

    def print_unique_by(self, iterable: Iterable, key=lambda r: r.identifier,
                        printer: Optional[Callable[[Record], None]] = None,
                        n: int = 10):
        printer = printer or self.print_details
        examples = []
        for record in iterable:
            if key(record) in examples:
                continue
            examples.append(key(record))
            printer(record)
            if len(examples) >= n:
                break

    @cached_property
    def most_sure(self):
        most_sure = map(lambda t: max(t[1], key=Record.confidence),
                        group_by(lambda r: r.identifier, self.correct).items())
        return sorted(most_sure, key=Record.confidence, reverse=True)

    @cached_property
    def most_wrong(self):
        most_wrong = map(lambda t: max(t[1], key=Record.confidence),
                         group_by(lambda r: r.identifier, self.wrong).items())
        return sorted(most_wrong, key=Record.confidence, reverse=True)

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
        self.print_unique_by(random_from(self.correct))

    @pickable_option
    def show_wrong_random(self):
        print("Wrong, random:")
        self.print_unique_by(random_from(self.wrong))

    @pickable_option
    def predicted_types_stats(self):
        print("Numbers of predicted types:")
        pprint(walk_values(len, dict(group_by(Record.most_likely,
                                              self.records))))

    @pickable_option
    def correct_predicted_types_stats(self):
        print("Numbers of predicted types in correct records:")
        pprint(walk_values(len, dict(group_by(Record.most_likely,
                                              self.correct))))

    @pickable_option
    def wrong_predicted_types_stats(self):
        print("Numbers of predicted types in wrong records:")
        pprint(walk_values(len, dict(group_by(Record.most_likely,
                                              self.wrong))))

    @pickable_option
    def show_accuracy(self):
        accuracy = len(self.correct) / len(self.records)
        real_accuracy\
            = (len(self.correct)
               / ilen(r for r in self.records if r.label in self.vocab))
        print("Accuracy: {}%, real accuracy: {}%"
              .format(100 * accuracy, 100 * real_accuracy))

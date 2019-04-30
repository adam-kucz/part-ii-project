import argparse
from enum import Enum, unique
from pathlib import Path
import sys
from typing import List, Callable, Iterable, NamedTuple, Tuple

from funcy import (identity, cached_property, collecting,
                   cut_prefix, compose, caller, walk_values,
                   partition, partial, decorator)
from numpy import percentile, std, mean

from ..util import csv_read


@unique
class RecordMode(Enum):
    IDENTIFIER = 'identifier'
    CONTEXT = 'context'
    OCCURENCES = 'occurence'

    def __str__(self):
        return self.value


class Record(NamedTuple):
    identifier: str
    inputs: Tuple
    label: str


# def with_pickable_options(cls):
#     if not hasattr(cls, 'options'):
#         cls.options = []
#     cls.options.extend(filter(rpartial(hasattr, 'pickable_option'),
#                               cls.__dict__.values()))

#     @monkey(cls)
#     def show_options(self):
#         for i, option in enumerate(self.options):
#             print("* [{}] {}".format(i, option.__name__))

#     @monkey(cls)
#     def pick(self, num, *args, **kwargs):
#         if 0 <= num < len(self.options):
#             return self.options[num](self, *args, **kwargs)
#         return num
#     return cls


class Pickable:
    @cached_property
    def options(self) -> List[Callable]:
        return [attr for attr in (getattr(self, name) for name in dir(self)
                                  if name != 'options')
                if hasattr(attr, 'pickable_option')]

    def show_options(self):
        print(f"Options for {self.__class__.__name__}")
        for i, option in enumerate(self.options):
            print("* [{}] {}".format(i, option.__name__))

    def pick(self, num, *args, **kwargs):
        if 0 <= num < len(self.options):
            return self.options[num](*args, **kwargs)
        return num


def pickable_option(func):
    func.pickable_option = True
    return func


class Interactive(Pickable):
    def main(self):
        self.show_options()
        choice = int(input("Choice: "))
        result = self.pick(choice)
        print()
        return result

    def loop(self):
        value = self.main()
        while value != -1:
            value = self.main()
        return value

    @pickable_option
    def exit(self):
        return -1


def _get_keyword_arg_name(*names: str) -> str:
    for name in names:
        if not name.startswith('-'):
            return name
        if name.startswith('--'):
            return cut_prefix(name, '--')
    raise ValueError("Name not found", names)


def cli_or_interactive(claz, params):
    kwargs = {}
    parser = argparse.ArgumentParser()
    parser_transforms = {}
    for names, opts in params.items():
        kw_arg_name = (opts.pop('argname', None)
                       or _get_keyword_arg_name(*names))
        transform = opts.pop('transform', identity)

        parser.add_argument(*names, **opts)
        parser_transforms[kw_arg_name]\
            = (lambda namespace, kw_arg_name=kw_arg_name, transform=transform:
               transform(getattr(namespace, kw_arg_name)))

        help_lines = ()
        prompt_name = kw_arg_name.capitalize().replace('_', ' ')
        if 'help' in opts:
            help_lines += f"*** {prompt_name} ***",
            help_lines += tuple(opts['help'].splitlines())
        default = (f" (default: {opts['default']})" if 'default' in opts
                   else '')
        prompt = f"{prompt_name}{default}: "
        full_transform = compose(transform, opts.get('type', str))
        if 'default' in opts:
            default = opts['default']
            full_transform = compose(full_transform,
                                     lambda s, d=default: s or d)

        def getter(help_lines=help_lines, prompt=prompt,
                   full_transform=full_transform):
            for line in help_lines:
                print(line)
            return full_transform(input(prompt))
        kwargs[kw_arg_name] = getter

    def interactive(cls):
        return cls(**walk_values(caller(), kwargs))
    claz.interactive = classmethod(interactive)

    def from_cli(cls):
        args = parser.parse_args()
        return cls(**walk_values(caller(args), parser_transforms))
    claz.from_cli = classmethod(from_cli)


@collecting
def read_dataset(mode: RecordMode, ctx_size: int, path: Path)\
        -> Iterable[Record]:
    raw = csv_read(path)
    if not raw:
        return
    assert len(raw[0]) % 2 == 0,\
        f"Invalid record length: {len(raw[0])}, record {raw[0]}"
    ctx_size = (ctx_size if mode == RecordMode.OCCURENCES
                else (len(raw[0]) - 2) // 2)
    for row in raw:
        inputs = row[:-1]
        if mode == RecordMode.OCCURENCES:
            inputs = partition(ctx_size, inputs)
        yield Record(identifier=row[ctx_size],
                     inputs=tuple(inputs),
                     label=row[-1])


def summarise_numbers(data: Iterable[float]) -> Tuple[float, float]:
    data = list(data)
    return (mean(data), std(data),
            *map(partial(percentile, data), (0, 25, 50, 75, 100)))


def redirect_stdout(filepath: Path):
    @decorator
    def redirecter(call):
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        with filepath.open('w') as fout:
            old_stdout = sys.stdout
            sys.stdout = fout
            call()
            sys.stdout = old_stdout
    return redirecter

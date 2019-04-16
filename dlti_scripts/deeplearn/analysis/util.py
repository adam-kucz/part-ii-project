import argparse
from typing import List, Callable

from funcy import (monkey, rpartial, identity,
                   cut_prefix, compose, caller, walk_values)


def with_pickable_options(cls):
    if not hasattr(cls, 'options'):
        cls.options = []
    cls.options.extend(filter(rpartial(hasattr, 'pickable_option'),
                              cls.__dict__.values()))

    @monkey(cls)
    def show_options(self):
        for i, option in enumerate(self.options):
            print("* [{}] {}".format(i, option.__name__))

    @monkey(cls)
    def pick(self, num, *args, **kwargs):
        if 0 <= num < len(self.options):
            return self.options[num](self, *args, **kwargs)
        return num
    return cls


def pickable_option(func):
    func.pickable_option = True
    return func


@with_pickable_options
class Interactive:
    options: List[Callable]

    def main(self):
        self.show_options()
        choice = int(input("Choice: "))
        result = self.pick(choice)
        print()
        return result

    def loop(self):
        value = self.main()
        while value != -1:
            self.main()
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
        if 'help' in opts:
            help_lines += f"*** {kw_arg_name.capitalize()} ***",
            help_lines += tuple(opts['help'].splitlines())
        default = (f" (default: {opts['default']})" if 'default' in opts
                   else '')
        prompt = f"{kw_arg_name.capitalize()}{default}: "
        full_transform = compose(transform, opts.get('type', str))

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

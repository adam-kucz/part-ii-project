from typing import Generator, List


def f(a: str, b: List[float])\
        -> Generator[int, None, None]:
    for i in range(5):  # type: int
        j: int = i * 2 + 3
        for string in ['abc', 'def']:  # type: str
            yield j

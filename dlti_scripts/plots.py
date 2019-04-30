from ast import literal_eval
from collections import defaultdict, OrderedDict, Counter
from pathlib import Path
from typing import Mapping, Set, List, Optional

from funcy import (compose, post_processing, walk_values, partial,
                   lmap, compact, re_find, re_test, lkeep, cat, lfilter, map,
                   select_keys)
import matplotlib.pyplot as plt
import numpy as np
from parse import parse
from sklearn.metrics import r2_score

from preprocessing.util import csv_read

PROJPATH = Path("/home/acalc79/synced/part-ii-project/")
OUTPATH = PROJPATH.joinpath("out")
DATAPATH = PROJPATH.joinpath("data")
LOGPATH = PROJPATH.joinpath("logs")

# SMALL_SIZE = 7
# MEDIUM_SIZE = 9
# BIGGER_SIZE = 10
# SCALE = 6

# controls default text sizes
# plt.rc('font', size=SMALL_SIZE * SCALE)
# fontsize of the axes title
# plt.rc('axes', titlesize=SMALL_SIZE * SCALE)
# fontsize of the x, y labels
# plt.rc('axes', labelsize=MEDIUM_SIZE * SCALE)
# fontsize of the tick labels
# plt.rc('xtick', labelsize=SMALL_SIZE * SCALE)
# fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE * SCALE)
# legend fontsize
# plt.rc('legend', fontsize=SMALL_SIZE * SCALE)
# fontsize of the figure title
# plt.rc('figure', titlesize=BIGGER_SIZE * SCALE)


def figure():
    return plt.figure(figsize=(8, 5))
    # return plt.figure(figsize=(8 * SCALE, 5 * SCALE))


def savetopdf(imgdir: Path, fig, name: str, title: Optional[str] = None):
    if not title:
        title = name.replace('-', ' ').replace('_', ' ').split(' ')
        title = ' '.join(word.capitalize() for word in title)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    if not imgdir.exists():
        imgdir.mkdir(parents=True)
    plt.savefig(imgdir.joinpath(name).with_suffix(".pdf"), format="pdf")


@post_processing(dict)
def from_logfile(logpath: Path) -> Mapping[str, Set[str]]:
    for line in logpath.read_text().split('\n'):
        if line:
            lineformat: str = "{split} ({}%): {projs}"
            parsed = parse(lineformat, line)
            if not parsed:
                raise ValueError("Line {} does not conform to format {}"
                                 .format(line, lineformat))
            yield parsed['split'], literal_eval(parsed['projs'])


def project_sizes_histogram(datapath, targetpath, splitpath):
    projs: Mapping[str, int] = {}
    for proj in filter(Path.is_dir, datapath.glob("*")):
        projs[proj.name] = sum(map(compose(len, csv_read), proj.glob("*.csv")))

    counts = walk_values(partial(lmap, projs.__getitem__),
                         from_logfile(splitpath))
    fig = figure()
    plt.hist(counts.values(), label=counts,
             color=('white', 'gray', 'black'),
             edgecolor='black', linewidth=0.5,
             bins=50, stacked=True)
    fig.legend(borderaxespad=2)
    plt.xlabel("# records")
    plt.ylabel("# projects")
    savetopdf(targetpath, fig, "project-count-histogram")


def type_counts_histogram(raw_dataname: str, dataset_name: str):
    datapath = DATAPATH.joinpath("raw", raw_dataname)
    common: Mapping[str, int] = {}
    unique: Mapping[str, int] = {}
    for proj in filter(Path.is_dir, datapath.glob("*")):
        typs = Counter(map(-1, cat(map(csv_read, proj.glob("*.csv")))))
        for typ, count in typs.items():
            if typ in common:
                common[typ] += count
            elif typ in unique:
                common[typ] = unique[typ] + count
                del unique[typ]
            else:
                unique[typ] = count

    dataset_path = DATAPATH.joinpath("sets", dataset_name)
    vocab = set(map(0, csv_read(dataset_path.joinpath("vocab.csv"))))
    fig = figure()
    _, _, bars = plt.hist(
        lmap(list, ((v for k, v in common.items() if k in vocab),
                    (v for k, v in unique.items() if k in vocab),
                    (v for k, v in common.items() if k not in vocab),
                    (v for k, v in unique.items() if k not in vocab))),
        label=('in-vocabulary, common', 'in-vocabulary, project-specific',
               'common', 'project-specific'),
        color=['0', '0.33', '0.66', '1'],
        edgecolor='black', linewidth=0.5,
        bins=np.logspace(np.log10(0.8), np.log10(30000), 30), log=True)
    # for barcontainer, pattern in zip(bars, ('', '', '', '////')):
    #     for patch in barcontainer:
    #         patch.set_hatch(pattern)
    plt.gca().set_xscale("log")
    plt.ylim(0.8, 5000)
    # start, end = plt.xlim()
    # start = np.round(start, -3)
    # start += 2000 - (start % 2000)
    # plt.xticks(np.arange(start, end, 2000))
    plt.xlabel("# times the type repeats")
    plt.ylabel("# types")
    fig.legend(borderaxespad=2)
    savetopdf(dataset_path, fig, "type-count-histogram")


def accuracies_plot(outpath: Path, logpath: Path, targetpath: Path):
    stats: Mapping[str, Mapping[str, List[float]]] = defaultdict(dict)

    identifier = r"([a-z_\d]+)\s*"
    num = r"([\d.]+)\s*"
    plist = fr"(\[[\d.]+(?:, [\d.]+)*\])\s*"
    pattern = fr"{identifier}:\s*{num}\+-\s*{num}{plist}"
    for net_path in filter(Path.is_dir, outpath.iterdir()):
        for run_path in filter(Path.is_dir, net_path.iterdir()):
            if run_path.stem != "run0":
                continue
            summary_path = run_path.joinpath(logpath)
            if not summary_path.is_file():
                continue
            for line in compact(summary_path.read_text().split('\n')):
                assert re_test(pattern, line), line
                stat_name, mean, std, percentiles = re_find(pattern, line)
                case_name = net_path.stem + "--" + run_path.stem
                stats[stat_name][case_name] = [float(mean), float(std),
                                               *literal_eval(percentiles)]

    for stat in ('accuracy', 'real_accuracy', 'real_top5', 'real_top3'):
        cases = sorted(stats[stat])
        fig = figure()
        labels = [re_find(r"[a-z]+(?:-\d+)?", name).replace('-', ' ')
                  for name in cases]
        # plt.errorbar(x=[x + 1.1 for x in range(len(cases))],
        #              y=[stats[stat][name][0] for name in cases],
        #              yerr=[stats[stat][name][1] for name in cases],
        #              capsize=8)
        plt.boxplot([stats[stat][name][2:] for name in cases],
                    labels=labels, whis=1000, medianprops={'color': 'black'})
        plt.ylim(0, 1)
        savetopdf(targetpath, fig, stat)


def accuracy_vs_coverage(outpath: Path, logpath: Path, targetpath: Path):
    for net_path in filter(Path.is_dir, outpath.iterdir()):
        for run_path in filter(Path.is_dir, net_path.iterdir()):
            summary_path = run_path.joinpath(logpath)
            if not summary_path.is_file():
                continue
            lines = lkeep(summary_path.read_text().split('\n'))
            if not lines:
                continue
            names = lines[0].split(',')[1:]
            stats: Mapping[str, Mapping[str, float]]\
                = OrderedDict((name, OrderedDict())
                              for name in names)
            for line in lines[1:]:
                values = line.split(',')
                test_name = values[0]
                assert len(values[1:]) == len(stats.values()),\
                    f"{values[1:]}, {stats.values()}"
                for val, mapping in zip(values[1:], stats.values()):
                    if val:
                        mapping[test_name] = float(val)
            scatterplot(stats, 'coverage', 'accuracy', run_path, targetpath)
            scatterplot(stats, 'coverage', 'real_accuracy',
                        run_path, targetpath)


def scatterplot(stats: Mapping[str, Mapping[str, float]],
                stat1: str, stat2: str, run_path: Path, targetpath: Path):
    net_stem = run_path.parent.stem
    case_name = f"{net_stem}-{run_path.stem}"
    fig = figure()
    all_projs = set(cat(map(dict.keys, stats.values())))
    projs = lfilter(lambda p: p in stats[stat1] and p in stats[stat2],
                    all_projs)
    x = lmap(stats[stat1].__getitem__, projs)
    y = lmap(stats[stat2].__getitem__, projs)
    plt.scatter(x=x, y=y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    r2 = r2_score(y, p(x))
    plt.plot(x, p(x), "k--", label=f"R^2 = {r2:4.2f}")
    plt.legend()
    components = net_stem.capitalize().split('-')
    title = components[0]
    if len(components) > 1 and re_test(r"\d+", components[1]):
        title += f"-{components[1]}"

    def pretty(stat):
        return stat.capitalize().replace('_', ' ')
    plt.xlabel(pretty(stat1))
    plt.ylabel(pretty(stat2))
    savetopdf(targetpath, fig, f"{case_name}-{stat1}-vs-{stat2}",
              f"{title} {pretty(stat1)} vs {pretty(stat2)}")


if __name__ == '__main__':
    project_sizes_histogram(DATAPATH.joinpath("raw", "identifier-f"),
                            DATAPATH.joinpath("raw"),
                            LOGPATH.joinpath("data-split.txt"))
    type_counts_histogram("identifier-f", "identifier-f-very-fine")
    # accuracies_plot(OUTPATH, Path("test-summary.txt"), OUTPATH)
    # accuracy_vs_coverage(OUTPATH, Path("test-stat.txt"),
    #                      OUTPATH.joinpath("figures"))

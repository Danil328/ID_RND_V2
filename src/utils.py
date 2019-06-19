from collections import defaultdict
from glob import glob


def rank_average(glob_files, loc_outfile):
    """
    https://gist.github.com/nlpjoe/38957f59cd5e6c6f8de7040a3bb9c486
    :param glob_files:
    :param loc_outfile:
    :return:
    """
    with open(loc_outfile, "w") as outfile:
        all_ranks = defaultdict(list)
        for i, glob_file in enumerate(glob(glob_files)):
            file_ranks = []
            print("parsing: {}".format(glob_file))
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate(lines):
                if e == 0 and i == 0:
                    outfile.write(line)
                elif e > 0:
                    r = line.strip().split(",")
                    file_ranks.append((float(r[1]), e, r[0]))
            for rank, item in enumerate(sorted(file_ranks)):
                all_ranks[(item[1], item[2])].append(rank)
        average_ranks = []
        for k in sorted(all_ranks):
            average_ranks.append((sum(all_ranks[k]) / len(all_ranks[k]), k))
        ranked_ranks = []
        for rank, k in enumerate(sorted(average_ranks)):
            ranked_ranks.append((k[1][0], k[1][1], rank / (len(average_ranks) - 1)))
        for k in sorted(ranked_ranks):
            outfile.write("%s,%s\n" % (k[1], k[2]))
        print("wrote to {}".format(loc_outfile))

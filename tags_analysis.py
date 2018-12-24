import argparse

def get_tag(line):
    sp_line = line.split(",")
    fname = sp_line[0]
    tags = []
    for el in sp_line:
        try:
            tags.append(int(el))
        except (ValueError, TypeError):
            pass
    return fname, tags


def get_labels(labels):
    tags = []
    names = []
    for i in labels:
        fname, tag = get_tag(i)
        tags.append(tag)
        names.append(fname)
    return names, tags


def get_unique_tag(tags, tag):
    tag_list = []
    for i in tags:
        if tag in i:
            tag_list.append(i)
        else:
            continue
    return tag_list


def tag_frequency(tags_list, main_tag, nlargest, n_classes=63):
    keys = [x for x in range(n_classes)]
    d = dict.fromkeys(keys)
    for i in d.keys():
        d[i] = sum(x.count(i) for x in tags_list)
    if main_tag in d:
        del d[main_tag]

    topk = sorted(d.items(), key=lambda item: item[1], reverse=True)[:nlargest]
    out = {main_tag: topk}
    return d, out


def count_tags(tags, n_classes=63):
    res = []
    for i in range(n_classes):
        tag_list = get_unique_tag(tags, i)
        _, out = tag_frequency(tag_list, i, args.nlargest)
        res.append(out)
    return res


def main(args):
    with open(args.ann_file) as infile:
        labels = infile.readlines()
    infile.close()

    names, tags = get_labels(labels)
    full = count_tags(tags)
    with open(args.out_dir, 'w') as file_handler:
        for item in full:
            file_handler.write("{}\n".format(item))
    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tags_analysis")
    parser.add_argument('--ann_file', type=str, metavar='PATH', help="path to the annotation file")
    parser.add_argument('--out_dir', type=str, metavar='PATH', help="path to the annotation file")
    parser.add_argument('--nlargest', type=int, default=5)
    args = parser.parse_args()
    main(args)

python tags_analysis.py --ann_file short_video_trainingnset_annotations.txt --out_dir mnt/ssd1/dataser/train_tags_analysis
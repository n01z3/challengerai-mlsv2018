#import system stuff
import sys
import argparse
import os.path as osp
import numpy as np

tags = ['dog', 'cat', 'mouse', 'rabbit', 'bird', 'scenery', 'local customs',
'dessing', 'baby', 'selfie - male', 'selfie - female', 'dessert making',
'seafood making', 'streetside snacks', 'drinks', 'hot pot', 'claw crane',
'handsign dance', 'street dance', 'international dance', 'pole dance', 
'ballet', 'square dancing', 'folk dance', 'drawing', 'handwriting', 'latter art',
'sand drawing', 'slime', 'origami', 'knitting', 'hair accessory', 'pottery',
'phone case', 'drums', 'guitar', 'piano', 'guzheng', 'violin', 'cello',
'hulusi', 'singing', 'games', 'entertainment', 'animation' , 'word art voicing',
'yoga', 'fitness', 'skateboard', 'basketball', 'parkour', 'diving', 'billiards',
'football', 'badminton', 'table tennis', 'brow painting', 'eyeliner', 'skincare',
'lipgloss', 'makeup removal', 'nail cosmetic', 'hair cosmetic']

def main(args):
    files = 0
    multi_label = 0 
    working_dir = osp.dirname(osp.abspath(__file__))
    labels = np.zeros(63)
    with open(args.ann_file) as infile, open(osp.join(working_dir, "stats.txt"), "w") as outfile:
        for line in infile:
            sp_line = line.split(",")
            #drop files that doesnt exist
            filepath = osp.join(args.train_dir, sp_line[0])
            
            if not osp.exists(filepath):
                continue
            
            if len(sp_line) > 2:
                multi_label += 1
            for el in sp_line:
                try:
                     labels[int(el)] += 1
                except (ValueError, TypeError):
                    files += 1
        for i in range(len(labels)):
            #print("label: {} | tag: {} | files: {}".format(i, tags[i], labels[i]))
            print("| {} | {} | {} |".format(i, tags[i], labels[i]))
            outfile.write("label: {} | tag: {} | files: {} \n".format(i, tags[i], labels[i]))
        outfile.write("amount of files: {}, files with more than 1 label: {} ".format(files, multi_label))
        outfile.flush() 
        outfile.close()
if __name__ == '__main__':
    #parameters?
    parser = argparse.ArgumentParser(description="collect the labels statistic ")
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--train_dir', type=str, metavar='PATH', help = "path to the train folder")
    main(parser.parse_args())
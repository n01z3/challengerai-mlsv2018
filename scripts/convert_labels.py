#import system stuff
import sys
import argparse
import os.path as osp
import numpy as np
import cv2

from torchvision.utils import * 

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
    labels = np.zeros(63)
    #makedir_exist_ok(args.out_dir)
    with open(args.ann_file) as infile, open(args.out_file, "w") as outfile:
        for line in infile:
            sp_line = line.split(",")

            file_name = line.split(",")[0]
            filepath = osp.join(args.data_dir, file_name)
            if osp.exists(filepath):
                cap = cv2.VideoCapture(filepath)
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                if len(sp_line) > 2:
                    multi_label += 1
                for el in sp_line:
                    try:
                        label = int(el)
                        labels[label] += 1
                        outfile.write('{},{},{}\n'.format(sp_line[0], label, int(n_frames)))
                        outfile.flush()
                    except (ValueError, TypeError):
                        files += 1
        print('stats: ')
        for i in range(len(labels)):
            print("label: {} tag: {} amount: {}".format(i, tags[i], labels[i]))
        print('files: {} multi-lable: {} '.format(files, multi_label))
        outfile.flush() 
        outfile.close()
        infile.close()

if __name__ == '__main__':
    #parameters?
    parser = argparse.ArgumentParser(description="re-create labels ")
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--data_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('--out_file', type=str, metavar='PATH', help = "path to the new annotation file")
    main(parser.parse_args())
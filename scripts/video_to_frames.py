#split video files into frames and save it as jpeg
#import system stuff
import sys
import argparse
import os.path as osp
import os
import cv2
import errno

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def main(args):

    ann_out_file = osp.join(osp.split(args.out_dir)[0], 'new_{}.txt'.format(osp.basename(args.ann_file)))
    mkdir_if_missing(args.out_dir)
    n_file = 0
    with open(args.ann_file) as infile, open(ann_out_file, "w") as outfile:
        for line in infile:
            sp_line = line.split(",")
            file_name = sp_line[0]
            filepath = osp.join(args.train_dir, file_name)
            if osp.exists(filepath):
                n_file = +1
                cap = cv2.VideoCapture(filepath)
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.f_frame)
                #frames = 0
                for i in range(args.n_frames):
                    ret, frame = cap.read()
                    out_filepath = osp.join(args.out_dir, '{}_{}.jpg'.format(osp.splitext(file_name)[0], i))
                    cv2.imwrite(out_filepath, frame)
                    #save file info
                    outfile.write('{}_{}.jpg'.format(osp.splitext(file_name)[0], i))
                    for k in range(1, len(sp_line)):
                        outfile.write(',{}'.format(sp_line[k]))
                    #outfile.write('\n')
                outfile.flush()
                cap.release()
                

        outfile.close()

if __name__ == '__main__':
    #parameters?
    parser = argparse.ArgumentParser(description="collect the labels statistic ")
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--train_dir', type=str, metavar='PATH', help = "path to the train folder")
    parser.add_argument('--out_dir', type = str, metavar='PATH', help = "path to the output folder")
    

    parser.add_argument('--f_frame', type = int, default = 0, help = "from which frame begin to split")
    parser.add_argument('--n_frames', type  = int, default = 1, help = "how many frames we want to extract")

    main(parser.parse_args())
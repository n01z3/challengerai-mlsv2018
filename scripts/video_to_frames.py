#split video files into frames and save it as jpeg
#import system stuff
import sys
import argparse
import os.path as osp
import os
import cv2
import errno

tags = ['dog', 'cat', 'mouse', 'rabbit', 'bird', 'scenery', 'local_customs',
'dessing', 'baby', 'selfie_male', 'selfie_female', 'dessert_making',
'seafood_making', 'streetside_snacks', 'drinks', 'hot_pot', 'claw_crane',
'handsign_dance', 'street_dance', 'international_dance', 'pole_dance', 
'ballet', 'square_dancing', 'folk_dance', 'drawing', 'handwriting', 'latter_art',
'sand_drawing', 'slime', 'origami', 'knitting', 'hair_accessory', 'pottery',
'phone_case', 'drums', 'guitar', 'piano', 'guzheng', 'violin', 'cello',
'hulusi', 'singing', 'games', 'entertainment', 'animation' , 'word_art_voicing',
'yoga', 'fitness', 'skateboard', 'basketball', 'parkour', 'diving', 'billiards',
'football', 'badminton', 'table_tennis', 'brow_painting', 'eyeliner', 'skincare',
'lipgloss', 'makeup_removal', 'nail_cosmetic', 'hair_cosmetic']

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def main(args):

    ann_out_file = osp.join(osp.splitext(args.out_dir)[0], 'new_{}'.format(osp.basename(args.ann_file)))
    mkdir_if_missing(args.out_dir)
    n_file = 0
    with open(args.ann_file) as infile, open(ann_out_file, "w") as outfile:
        for line in infile:
            sp_line = line.split(",")
            file_name = sp_line[0]
            filepath = osp.join(args.data_dir, file_name)
            if osp.exists(filepath):
                n_file = +1
                cap = cv2.VideoCapture(filepath)
                #frames = 0
                #labels 
                file_name = osp.splitext(file_name)[0]
                for k in range(1, len(sp_line)):
                    label = int(sp_line[k])

                    #create folder for specific label
                    mkdir_if_missing(osp.join(args.out_dir, '{}_{}'.format(tags[label], label)))
                    #dump video
                    n_frame = 0
                    while True:
                        ret, frame = cap.read()
                        if ret == True:
                            if n_frame % args.th_frame == 0:
                                out_filepath = osp.join(args.out_dir, '{}_{}'.format(tags[label], label), '{}_{}.jpg'.format(file_name, n_frame))
                                cv2.imwrite(out_filepath, frame)
                                outfile.write('{}/{}_{}.jpg'.format('{}_{}'.format(tags[label], label), file_name, n_frame))
                                outfile.write(',{}\n'.format('{}'.format(label)))
                                outfile.flush()
                            n_frame += 1
                        else:
                            cap.release()
                            break
                if n_file % 10 == 0:
                    print ('{} files is processed'.format(n_file))

        outfile.close()

if __name__ == '__main__':
    #parameters?
    parser = argparse.ArgumentParser(description="split video to frames ")
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--data_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('--out_dir', type = str, metavar='PATH', help = "path to the output folder")

    parser.add_argument('--th_frame', type = int, help = "extract each Nth frame", default = 4)
    
    main(parser.parse_args())
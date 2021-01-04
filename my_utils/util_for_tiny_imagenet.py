import argparse
import os
import shutil
from glob import glob


'''
[Quick Overview of this source code]
train                             ➜ train
 ↳ n01443537                        ↳ n01443537
   ↳ images                             ↳ *.JPEG
     ↳ *.JPEG                       ↳ n01629819
 ↳ n01629819                            ↳ *.JPEG
   ↳ images                         ...
     ↳ *.JPEG
 ...

val                               ➜ valid
 ↳ images                           ↳ n01443537
   ↳ *.JPEG                             ↳ *.JPEG
 ↳ val_annotations.txt              ↳ n01629819
                                        ↳ *.JPEG
                                    ...
'''

parser = argparse.ArgumentParser(description='Tiny Imagenet Util (For Classification)')
parser.add_argument('--dir', type=str)

parser.dir

import os
import sys
import shutil

# hyper-params ##############################################################################################################
cifar100_dir = './cifar-100'
sifar_dir = 'SIFAR-A'
#############################################################################################################################

if sifar_dir == 'SIFAR-A':
    alphas = {'0.1': ['bicycle', 'beaver', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.2': ['bicycle', 'bus', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.3': ['bicycle', 'bus', 'motorcycle', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.4': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.5': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.6': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'bed', 'bee', 'camel', 'baby'],
              '0.7': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'bee', 'camel', 'baby'],
              '0.8': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'camel', 'baby'],
              '0.9': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'baby'],
              '1.0': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}
elif sifar_dir == 'SIFAR-B':
    alphas = {'0.1': ['beaver', 'maple_tree', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.2': ['beaver', 'dolphin', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.3': ['beaver', 'dolphin', 'otter', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.4': ['beaver', 'dolphin', 'otter', 'seal', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.5': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.6': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'bed', 'bee', 'camel', 'baby'],
              '0.7': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'bee', 'camel', 'baby'],
              '0.8': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'camel', 'baby'],
              '0.9': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'baby'],
              '1.0': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']}
elif sifar_dir == 'SIFAR-C':
    alphas = {'0.1': ['fox', 'maple_tree', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.2': ['fox', 'porcupine', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.3': ['fox', 'porcupine', 'possum', 'bottle', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.4': ['fox', 'porcupine', 'possum', 'raccoon', 'apple', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.5': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'clock', 'bed', 'bee', 'camel', 'baby'],
              '0.6': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'bed', 'bee', 'camel', 'baby'],
              '0.7': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'bee', 'camel', 'baby'],
              '0.8': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'rabbit', 'camel', 'baby'],
              '0.9': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'rabbit', 'shrew', 'baby'],
              '1.0': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']}


def check_all_classes_exist(sifar_dir, cifar100_dir):
    target_classes = []
    for alpha in alphas.keys():
        target_classes.extend(alphas[alpha])
    target_classes = set(target_classes)
    classes = os.listdir(f'{cifar100_dir}/test/')
    assert len(classes) == 100, f'check {cifar100_dir}!'
    for target_class in target_classes:
        assert target_class in classes, f'{target_class} does not exist in {cifar100_dir}!'
    print(f'check_all_classes_exist() for constructing "{sifar_dir}" ... ok!')


check_all_classes_exist(sifar_dir, cifar100_dir)

for alpha in alphas.keys():
    print(f'α = {alpha}', end=' ... ')
    current_classes = alphas[alpha]
    for mode in ['train', 'test']:
        for i, current_class in enumerate(current_classes):
            shutil.copytree(f'{cifar100_dir}/{mode}/{current_class}',
                            f'{sifar_dir}/{alpha}/{mode}/{i+1}-{current_class}')
    print('done!')

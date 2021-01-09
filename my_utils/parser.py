import argparse


class Parser:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='Pytorch Image Classification (github.com/unique-chan)')
        if mode == 'train':
            self.add_arguments_for_train()      # train - valid
        else:
            self.add_arguments_for_test()       # test

    def parse_args(self):
        return self.parser.parse_args()

    def add_arguments_for_train(self):
        self.parser.add_argument('--loss_func', default='ERM', type=str,
                                 help='loss function (default:ERM <cross-entropy>)')
        self.parser.add_argument('--network_name', type=str, help='network name')
        self.parser.add_argument('--dataset_dir', type=str, help='dataset path')
        self.parser.add_argument('--height', default=32, type=int, help='image height (default: 32)')
        self.parser.add_argument('--width', default=32, type=int, help='image height (default: 32)')
        self.parser.add_argument('--lr', default=0.1, type=float,
                                 help='initial learning rate (default: 0.1)')
        self.parser.add_argument('--epochs', default=5, type=int, help='epochs (default: 5)')
        self.parser.add_argument('--batch_size', default=128, type=int, help='batch_size (default: 128)')
        self.parser.add_argument('--lr_step', default=[100, 150], type=list,
                                 help='learning rate step decay milestones (default: [100, 150])')
        self.parser.add_argument('--lr_step_gamma', default=0.1, type=float,
                                 help='learning rate step decay gamma (default: 0.1)')
        self.parser.add_argument('--lr_warmup_epochs', default=5, type=int,
                                 help='epochs for learning rate warming up (default: 5)')
        self.parser.add_argument('--store', action='store_true',
                                 help='store the best trained model')
        self.parser.add_argument('--test', action='store_true',
                                 help='immediately test the model after training is done')
        self.parser.add_argument('--mean_std', action='store_true',
                                 help='initially normalize the entire data '
                                      'with the training mean and standard deviation')
        self.parser.add_argument('--clip', default=0, type=float,
                                 help='gradient clipping constant (default: 0) ')
        self.parser.add_argument('--center_crop_size', default=0, type=int,
                                 help='central cropping size for validation/test '
                                      '(default: 0 (No center crop)) (eg. 56 -> 56 x 56 central cropping)')

    def add_arguments_for_test(self):
        self.parser.add_argument('--datetime', type=str, help='datetime')

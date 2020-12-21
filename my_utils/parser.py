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
        self.parser.add_argument('--lr_step', default=[30, 50, 80, 100], type=list,
                                 help='learning rate step decay milestones (default: [30, 50, 80, 100])')
        self.parser.add_argument('--lr_step_gamma', default=0.2, type=float,
                                 help='learning rate step decay gamma (default: 0.2)')
        self.parser.add_argument('--lr_warmup', action='store_true',
                                 help='initial learning rate warming up for first 5 epochs')
        self.parser.add_argument('--store', action='store_true',
                                 help='store the best trained model')
        self.parser.add_argument('--test', action='store_true',
                                 help='immediately test the model after training is done')

    def add_arguments_for_test(self):
        self.parser.add_argument('--datetime', type=str, help='datetime')

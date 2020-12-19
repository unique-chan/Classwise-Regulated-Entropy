from my_utils import parser, loader, model, trainer, util
from warnings import filterwarnings

if __name__ == '__main__':
    filterwarnings('ignore')

    # Random Seeds For Reproducibility
    util.fix_random_seed()

    # Parser
    my_parser = parser.Parser(mode='train')
    my_args = my_parser.parse_args()

    # Loader (Train / Valid)
    my_loader = loader.Loader(my_args.dataset_dir)
    my_train_loader = my_loader.get_train_loader(my_args.height, my_args.width, my_args.batch_size)
    my_valid_loader = my_loader.get_valid_loader(my_args.batch_size)
    my_test_loader = my_loader.get_test_loader(my_args.batch_size)

    # Model
    my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)

    # Train and validation
    my_trainer = trainer.Trainer(my_model, my_train_loader, my_args.lr, my_loader.num_classes, my_args.loss_func)
    for cur_epoch in range(0, my_args.epochs):
        my_trainer.train(cur_epoch, my_train_loader, my_args.lr_warmup)
        my_trainer.valid(cur_epoch, my_valid_loader)

    # Test
    if my_args.test:
        my_trainer.test(my_test_loader)

    # for debug, remove later!
    print('train_loss:', my_trainer.train_loss_list)
    print('valid_loss:', my_trainer.valid_loss_list)
    print('test_loss:', my_trainer.test_loss)

    print('train_acc:', my_trainer.train_top1_acc_list)
    print('valid_acc:', my_trainer.valid_top1_acc_list)
    print('test_acc:', my_trainer.test_top1_acc)

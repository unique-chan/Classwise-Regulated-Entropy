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
    my_model = model.model(my_args.network_name)

    # Train and validation
    my_trainer = trainer.Trainer(my_model, my_train_loader, my_args.lr)
    for cur_epoch in range(0, my_args.epochs):
        my_trainer.train(my_train_loader, my_args.lr_warmup)
        my_trainer.valid(my_valid_loader)

    if my_args.test:
        my_trainer.test(my_test_loader)

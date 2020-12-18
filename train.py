from my_utils import parser, loader, model, trainer

if __name__ == '__main__':
    # Parser
    my_parser = parser.Parser(mode='train')
    my_args = my_parser.parse_args()

    # Loader
    my_loader = loader.Loader(my_args.dataset_dir)
    my_train_loader = my_loader.get_train_loader(my_args.height, my_args.width, my_args.batch_size)
    # my_valid_loader = my_loader.get_valid_loader()

    # Model
    my_model = model.model(my_args.network_name)

    # Trainer
    my_trainer = trainer.Trainer(my_model, my_train_loader, my_args.lr)
    # for cur_epoch in range(0, my_args.epochs):
    for cur_epoch in range(0, 3):
        print('for debug... remove later >> cur_epoch', cur_epoch)
        my_trainer.train(cur_epoch, my_train_loader, my_args.lr_warmup)
        # my_trainer.valid(cur_epoch, my_valid_loader)

    print('end')
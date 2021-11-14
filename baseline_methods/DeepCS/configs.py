def config_JointEmbeder():
    conf = {
        'gpu_id': 0,

        'dataset_name': 'CodeSearchDataset',

        'train_name': 'train_name.h5',
        'train_tokens': 'train_token.h5',
        'train_api': 'train_api.h5',
        'train_desc': 'train_doc.h5',

        'val_name': 'val_name.h5',
        'val_tokens': 'val_token.h5',
        'val_api': 'val_api.h5',
        'val_desc': 'val_doc.h5',

        'test_name': 'test_name.h5',
        'test_api': 'test_api.h5',
        'test_tokens': 'test_token.h5',
        'test_desc': 'test_doc.h5',

        'tokens_len': 50,
        'desc_len': 30,
        'name_len': 6,
        'api_len': 30,
        'n_words': 10000,

        'batch_size': 32,
        'nb_epoch': 200,
        'learning_rate': 0.0003,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
        'fp16': False,
        'fp16_opt_level': 'O1',

        'emb_size': 512,
        'n_hidden': 512,
        'lstm_dims': 256,

        'margin': 0.6,
        'sim_measure': 'cos',
        'dropout': 0.1,
    }
    return conf

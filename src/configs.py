def config_TranEmbeder():
    conf = {
        'gpu_id': 0,

        'code_nn': 'LSTM',
        'doc_nn': 'LSTM',
        'mode': 'Block',

        'tran_with_attention': 0,
        'doc_with_attention': 1,
        'tran_transform': 0,
        'doc_transform': 1,

        'transform_every_modal': 0,
        'transform_attn_out': 0,
        'use_tanh': 0,

        'emb_size': 512,
        'margin': 0.6,
        'sim_measure': 'cos',
        'dropout': 0.1,

        'batch_size': 32,
        'n_layers_LSTM': 1,
        'n_hidden': 512,

        'doc_len': 35,
        'tran_len': 1000,
        'tran_seq_len': 10,
        'tran_block_len': 100,

        'n_tran_words': 10000,
        'n_doc_words': 10000,
        'n_tran_doc_words': 15000,

        'dataset_name': 'TranCSDataset',
        'train_tran': 'train_tran.h5',
        'train_doc': 'train_doc.h5',

        'test_tran': 'test_tran.h5',
        'test_doc': 'test_doc.h5',

        'n_epoch': 200,
        'learning_rate': 0.0003,

        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
    }
    return conf

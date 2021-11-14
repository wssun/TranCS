def config_MultiEmbeder():
    conf = {
        'device': 0,
        'dataset_name': 'CodeSearchDataset',

        'train_tokens': 'train_token_complete.h5',
        'train_ast': 'train_ast.pkl',
        'train_cfg': 'train_cfg.json',
        'train_desc': 'train_doc.h5',

        'val_tokens': 'val_token_complete.h5',
        'val_ast': 'val_ast.pkl',
        'val_cfg': 'val_cfg.json',
        'val_desc': 'val_doc.h5',

        'test_tokens': 'test_token_complete.h5',
        'test_ast': 'test_ast.pkl',
        'test_cfg': 'test_cfg.json',
        'test_desc': 'test_doc.h5',

        'vocab_ast': 'ast.json',

        'tokens_len': 100,
        'desc_len': 30,
        'n_words': 10000,
        'n_ast_words': 10000,
        'n_node': 200,

        'max_word_num':35,

        'n_edge_types': 1,
        'state_dim': 512,
        'annotation_dim': 5,
        'n_layers': 1,
        'n_steps': 5,

        'treelstm_cell_type': 'nary',
        'output_type': 'no_reduce',

        'batch_size': 2,
        'nb_epoch': 200,

        'learning_rate': 0.0003,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
        'fp16': False,
        'fp16_opt_level': 'O1',

        'use_desc_attn': 1,
        'use_tanh': 1,
        'emb_size': 300,
        'n_hidden': 512,

        'margin': 0.6,
        'sim_measure': 'cos',

        'dropout': 0.1,
    }
    return conf

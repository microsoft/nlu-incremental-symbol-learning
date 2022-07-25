local data_dir = "test/data/";

{
  dataset_reader: {
    type: "calflow",
    use_program: true,
    use_context: true,
    source_token_indexers: {
      source_tokens: {
        type: "single_id",
        namespace: "source_tokens",
      },
      source_token_characters: {
        type: "characters",
        namespace: "source_token_characters",
        min_padding_length: 5,
      },
    },
    target_token_indexers: {
      target_tokens: {
        type: "single_id",
        namespace: "target_tokens",
      },
      target_token_characters: {
        type: "characters",
        namespace: "target_token_characters",
        min_padding_length: 5,
      },
    },
    generation_token_indexers: {
      generation_tokens: {
        type: "single_id",
        namespace: "generation_tokens",
      }
    },
    source_index_indexers: {
      source_indices: {
        type: "single_id",
        namespace: "source_indices_tokens",
      }
    },
    source_head_indexers: {
      source_edge_heads: {
        type: "single_id",
        namespace: "source_head_tokens",
      }
    },
    source_type_indexers: {
      source_edge_types: {
        type: "single_id",
        namespace: "source_type_tokens",
      }
    },
    line_limit: 2,
  },
  train_data_path: data_dir + "difficult",
  validation_data_path: data_dir + "difficult",
  test_data_path: null,
  datasets_for_vocab_creation: [
    "train"
  ],

  vocabulary: {
    non_padded_namespaces: [],
    min_count: {
      source_tokens: 1,
      target_tokens: 1,
      generation_tokens: 1,
    },
    max_vocab_size: {
      source_tokens: 18000,
      target_tokens: 12200,
      generation_tokens: 12200,
    },
  },

  model: {
    type: "calflow_parser",
    fxn_of_interest: "ChooseCreateEventFromConstraint",
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          embedding_dim: 300,
          trainable: true,
        },
        source_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "source_token_characters",
            embedding_dim: 16,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.00,
        },
      },
    },
    encoder_index_embedder: {
      token_embedders: {
        source_indices: {
          type: "embedding",
          vocab_namespace: "source_indices_tokens",
          embedding_dim: 16,
          trainable: true,
        },
      },
    },
    encoder_head_embedder: {
      token_embedders: {
        source_edge_heads: {
          type: "embedding",
          vocab_namespace: "source_head_tokens",
          embedding_dim: 16,
          trainable: true,
        },
      },
    },
    encoder_type_embedder:{
      token_embedders: {
        source_edge_types: {
          type: "embedding",
          vocab_namespace: "source_type_tokens",
          embedding_dim: 16,
          trainable: true,
        },
      },
    },
    encoder: {
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 300 + 50 + 3 * 16,
      hidden_size: 128,
      num_layers: 2,
      recurrent_dropout_probability: 0.00,
      use_highway: false,
    },
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          #pretrained_file: glove_embeddings,
          embedding_dim: 300,
          trainable: true,
        },
        target_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "target_token_characters",
            embedding_dim: 16,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.00,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 200,
      embedding_dim: 50,
    },
    decoder: {
      rnn_cell: {
        input_size: 300 + 50 + 50 + 256,
        hidden_size: 256,
        num_layers: 2,
        recurrent_dropout_probability: 0.01,
        use_highway: false,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 256,
        key_vector_dim: 256,
        output_vector_dim: 256,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 256,
          key_vector_dim: 256,
          hidden_vector_dim: 256, 
          use_coverage: false,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 256,
        key_vector_dim: 256,
        output_vector_dim: 256,
        attention: {
          type: "mlp",
          query_vector_dim: 256,
          key_vector_dim: 256,
          hidden_vector_dim: 256,
          use_coverage: false,
        },
      },
      dropout: 0.00,
    },
    extended_pointer_generator: {
      input_vector_dim: 256,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      query_vector_dim: 256,
      key_vector_dim: 256,
      edge_head_vector_dim: 256,
      edge_type_vector_dim: 128,
      attention: {
        type: "biaffine",
        query_vector_dim: 256,
        key_vector_dim: 256,
      },
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.0,
    beam_size: 2,
    max_decoding_steps: 50,
    target_output_namespace: "generation_tokens",
    edge_type_namespace: "edge_types",
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 64,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 32,
  },

  trainer: {
    type: "calflow_parsing",
    num_epochs: 250,
    warmup_epochs: 240,
    patience: 250,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: -1,
    num_serialized_models_to_keep: 1,
    validation_metric: "+exact_match",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
    },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "valid",
    validation_prediction_path: "valid_validation.txt",
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}

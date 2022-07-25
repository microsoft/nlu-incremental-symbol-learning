local data_dir = "test/data/smcalflow.agent.data/";

{
  dataset_reader: {
    type: "vanilla_calflow",
    use_agent_utterance: true,
    use_context: true,
    fxn_of_interest: "FindEventWrapperWithDefaults",
    source_token_indexers: {
      source_tokens: {
        type: "single_id",
        namespace: "source_tokens",
      },
    },
    target_token_indexers: {
      target_tokens: {
        type: "single_id",
        namespace: "target_tokens",
      },
    },
    generation_token_indexers: {
      generation_tokens: {
        type: "single_id",
        namespace: "generation_tokens",
      }
    },
  },
  train_data_path: data_dir + "tiny",
  validation_data_path: data_dir + "tiny",
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
    type: "vanilla_calflow_parser",
    fxn_of_interest: "FindEventWrapperWithDefaults",
    loss_weights: [0.1, 0.9],
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          embedding_dim: 300,
          trainable: true,
        },
      },
    },
    encoder: {
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 300 ,
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
      },
    },
    decoder: {
      rnn_cell: {
        input_size: 300 +  256,
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
      dropout: 0.00,
    },
    extended_pointer_generator: {
      input_vector_dim: 256,
      source_copy: true,
    },
    label_smoothing: {
        type: "no_sum",
        smoothing: 0.0,
    },
    dropout: 0.0,
    beam_size: 2,
    max_decoding_steps: 50,
    target_output_namespace: "generation_tokens",
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
    type: "vanilla_calflow_parsing",
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

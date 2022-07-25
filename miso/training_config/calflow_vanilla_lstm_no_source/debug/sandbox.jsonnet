#local data_dir = "/home/t-eliass/resources/data/smcalflow_samples/FindManager/10000_100/";
local data_dir = "/home/t-eliass/resources/data/smcalflow.agent.data/";
local glove_embeddings = "/home/t-eliass/resources/data/glove.840B.300d.zip";

{
  dataset_reader: {
    type: "vanilla_calflow",
    use_agent_utterance: true,
    use_context: true, 
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
    tokenizer: null,
  },
  train_data_path: data_dir + "train",
  validation_data_path: data_dir + "dev_valid",
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
      source_tokens: 200000,
      target_tokens: 122000,
      generation_tokens: 122000,
    },
  },

  model: {
    type: "vanilla_calflow_parser",
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          embedding_dim: 300,
          pretrained_file: glove_embeddings,
          trainable: true,
        },
      },
    },
    encoder: {
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 300,
      hidden_size: 192,
      num_layers: 2,
      recurrent_dropout_probability: 0.3,
      use_highway: false,
    },
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          pretrained_file: glove_embeddings,
          embedding_dim: 300,
          trainable: true,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 500,
      embedding_dim: 50,
    },
    decoder: {
      rnn_cell: {
        input_size: 300 + 50 + 384,
        hidden_size: 384,
        num_layers: 2,
        recurrent_dropout_probability: 0.3,
        use_highway: false,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 384,
        key_vector_dim: 384,
        output_vector_dim: 384,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 384,
          key_vector_dim: 384,
          hidden_vector_dim: 64, 
          use_coverage: false,
        },
      },
      dropout: 0.3,
    },
    extended_pointer_generator: {
      input_vector_dim: 384,
      source_copy: true,
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.3,
    beam_size: 5,
    max_decoding_steps: 100,
    target_output_namespace: "generation_tokens",
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 500,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 500,
    instances_per_epoch: 500,
  },
  trainer: {
    type: "vanilla_calflow_parsing",
    num_epochs: 250,
    warmup_epochs: 0,
    patience: 20,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 2,
    validation_metric: "+exact_match",
    optimizer: {
      type: "adam",
      lr: 0.001,
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

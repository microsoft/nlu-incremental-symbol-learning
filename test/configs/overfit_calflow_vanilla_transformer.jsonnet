local data_dir = "test/data/smcalflow.agent.data/";
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
    type: "vanilla_calflow_transformer_parser",
    fxn_of_interest: "ChooseCreateEventFromConstraint",
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          #pretrained_file: glove_embeddings,
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
          dropout: 0.01,
        },
      },
    },
    encoder: {
        type: "transformer_encoder",
        input_size: 300 + 50,
        hidden_size: 64,
        num_layers: 7,
        encoder_layer: {
            type: "pre_norm",
            d_model: 64,
            n_head: 16,
            norm: {type: "scale_norm",
                  dim: 64},
            dim_feedforward: 128,
            init_scale: 128
            },
        dropout: 0.0,
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
          dropout: 0.01,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 200,
      embedding_dim: 50,
    },
    decoder: {
      type: "transformer_decoder",
      input_size: 300 + 50 + 50,
      hidden_size: 64,
      num_layers: 4,
      use_coverage: false,
      decoder_layer: {
        type: "pre_norm",
        d_model: 64, 
        n_head: 4, 
        norm: {type: "scale_norm",
               dim: 64},
        dim_feedforward: 128,
        dropout: 0.2, 
        init_scale: 4,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 64,
        key_vector_dim: 64,
        output_vector_dim: 64,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 64,
          key_vector_dim: 64,
          hidden_vector_dim: 64, 
          use_coverage: false,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 64,
        key_vector_dim: 64,
        output_vector_dim: 64,
        attention: {
          type: "mlp",
          query_vector_dim: 64,
          key_vector_dim: 64,
          hidden_vector_dim: 64,
          use_coverage: false, 
        },
      },
    },
    extended_pointer_generator: {
      input_vector_dim: 64,
      source_copy: true,
      target_copy: true,
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.0,
    beam_size: 2,
    max_decoding_steps: 100,
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
    num_epochs: 550,
    warmup_epochs: 540,
    patience: 10050,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 1,
    validation_metric: "+exact_match",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
    },
    learning_rate_scheduler: {
       type: "noam",
       model_size: 64, 
       warmup_steps: 1000,
       factor: 1,
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

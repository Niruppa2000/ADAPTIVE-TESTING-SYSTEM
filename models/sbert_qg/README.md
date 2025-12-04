---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: On the way they were joined by thousands. with the British, as
    they had done in 1921-22, but also to break colonial laws. Thousands in different
    parts of the country broke the salt law, manufactured salt and demonstrated in
    front of government salt factories. As the movement spread, foreign cloth was
    boycotted, and liquor shops were picketed.
  sentences:
  - provincial interests
  - Thousands
  - Omprakash
- source_sentence: 'develop their own culture. Right to Constitutional Remedies: -
    If a politician in one state decides to not allow labourers This allows citizens
    to move the from other states to work in his state. court if they believe that
    any of their - If a group of people are not given permission to open a Fundamental
    Rights have been Telugu-medium school in Kerala. violated by the State.'
  sentences:
  - develop their own culture.
  - several photographs from his collection
  - can you imagine what it is to live in a society where the cultivation of crops
    was unknown?
- source_sentence: 'u Unit Pages: Each Unit begins with a Unit Page for teachers to
    help highlight the main points raised in the chapters. u Note on Evaluation: As
    with the Class VI text, this book does not contain definitions or a synthesis
    of concepts. While we recognise that this makes it difficult for teachers to evaluate
    what the child has learnt, our attempt is also to try and shift some of the understanding
    amongst teachers on what children are expected to learn and how such learning
    should be evaluated. This book contains a short note on evaluation procedures
    that we hope will assist teachers in their efforts to move students away from
    rote learning.'
  sentences:
  - Balkans also became the scene of big power rivalry
  - move students away from rote learning
  - Marx argued that industrial society was ‘capitalist’
- source_sentence: 'Bharatiya Janata Party (BJP) 282 Who will be present for Communist
    Party of India (CPI) 1 discussions in the Lok Sabha? Communist Party of India
    (Marxist) (CPM) 9 Is this process similar to what Indian National Congress (INC)
    44 you have read about in Nationalist Congress Party (NCP) 6 Class VII? State
    Parties (Regional Parties) Aam Aadmi Party (AAP) 4 All India Anna Dravida Munnetra
    Kazhagam 37 All India Trinamool Congress 34 All India United Democratic Front
    3 Biju Janata Dal (BJD) 20 Indian National Lok Dal (INLD) 2 Indian Union Muslim
    League (IUML) 2 Jammu and Kashmir Peoples Democratic Party 3 The photograph on   Janata
    Dal (Secular) 2 shows results from the 3rd Lok Janata Dal (United) 2 Sabha elections
    held in 1962. Jharkhand Mukti Morcha (JMM) 2 Use the photograph to answer Lok
    Jan Shakti Party 6 the following questions: Rashtriya Janata Dal (RJD) 4 a.'
  sentences:
  - '282'
  - 1,000 rupee note
  - She has a decision to make
- source_sentence: When they acted in the name of Mahatma Gandhi, or linked their
    movement to that of the Congress, they were identifying with a movement which
    went beyond the limits of their immediate locality. 5 – Chauri Chaura, 1922. At
    Chauri Chaura in Gorakhpur, a peaceful demonstration in a bazaar turned into a
    violent clash with the police. Hearing of the incident, Mahatma Gandhi called
    a halt to the Non-Cooperation Movement.
  sentences:
  - 272 All India Anna
  - in what way
  - Chauri Chaura in Gorakhpur
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'When they acted in the name of Mahatma Gandhi, or linked their movement to that of the Congress, they were identifying with a movement which went beyond the limits of their immediate locality. 5 – Chauri Chaura, 1922. At Chauri Chaura in Gorakhpur, a peaceful demonstration in a bazaar turned into a violent clash with the police. Hearing of the incident, Mahatma Gandhi called a halt to the Non-Cooperation Movement.',
    'Chauri Chaura in Gorakhpur',
    'in what way',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.5996, -0.0295],
#         [ 0.5996,  1.0000,  0.0113],
#         [-0.0295,  0.0113,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                       |
  |:--------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                           |
  | details | <ul><li>min: 31 tokens</li><li>mean: 97.44 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 9.72 tokens</li><li>max: 58 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
  | <code>n 14 OUR PASTS–I 2019–20 SSSSSiiiiittttteeeeesssss are places where the remains of things (tools, pots, buildings etc.) were found. These were made, used and left behind by people. These may be found on the surface of the earth, buried under the earth, or sometimes even under water. You will learn more about different sites in later chapters.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>n 14 OUR PASTS–I 2019–20 SSSSSiiiiiiittttttttteeeeeeesssss</code>                                      |
  | <code>Most African- American children can only afford to attend government schools that have fewer facilities and poorly qualified teachers as compared to white students who either go to private schools or live in areas where the government schools are as highly rated as private schools. Chapter 1: On Equality 13 2019-2020 Excerpt from Article 15 of the Indian Constitution Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth. (1) The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them. (2) No citizen shall, on grounds only of religion, race, caste, sex, place of birth or any of them, be subject to any disability, liability, restriction or condition with regard to – (a) access to shops, public restaurants, hotels and places of public entertainment; or (b) the use of wells, tanks, bathing ghats, roads and places of public resort maintained wholly or partly out of State funds or dedicat...</code> | <code>fewer facilities and poorly qualified teachers</code>                                                  |
  | <code>As tenants they had no security of tenure, being regularly evicted so that they could acquire no right over the leased land. The peasant movement demanded reduction of revenue, abolition of begar, and Activity social boycott of oppressive landlords. In many places nai – dhobi bandhs were organised by panchayats to deprive landlords of the If you were a peasant in Uttar Pradesh in 1920, services of even barbers and washermen. In June 1920, Jawaharlal how would you have responded to Gandhiji’s call for Swaraj?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>peasant movement demanded reduction of revenue, abolition of begar, and Activity social boycott</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 50
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 50
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch   | Step | Training Loss |
|:-------:|:----:|:-------------:|
| 7.9365  | 500  | 0.1335        |
| 15.8730 | 1000 | 0.014         |
| 23.8095 | 1500 | 0.0068        |
| 31.7460 | 2000 | 0.0053        |
| 39.6825 | 2500 | 0.0042        |
| 47.6190 | 3000 | 0.0043        |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.0+cu126
- Accelerate: 1.12.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
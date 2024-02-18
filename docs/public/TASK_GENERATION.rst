.. _generation-label:

Generation
==========

Models
------

+------------------------------------------------------------------+------+-----------+
| Model                                                            | Lang | Task      |
+==================================================================+======+===========+
| `timpal0l/mdeberta-v3-base-squad2                                | EN   | CLOSED QA |
| <https://huggingface.co/timpal0l/mdeberta-v3-base-squad2>`__     |      |           |
+------------------------------------------------------------------+------+-----------+
| `VMware/electra-small-mrqa                                       | EN   | CLOSED QA |
| <https://huggingface.co/VMware/electra-small-mrqa>`__            |      |           |
+------------------------------------------------------------------+------+-----------+
| `EleutherAI/pythia-160m-deduped                                  | EN   |  OPEN QA  |
| <https://huggingface.co/EleutherAI/pythia-160m-deduped>`__       |      |           |
+------------------------------------------------------------------+------+-----------+
| `JackFram/llama-68m                                              | EN   |  OPEN QA  |
| <https://huggingface.co/JackFram/llama-68m>`__                   |      |           |
+------------------------------------------------------------------+------+-----------+
| `EleutherAI/gpt-neo-125m                                         | EN   |  OPEN QA  |
| <https://huggingface.co/EleutherAI/gpt-neo-125m>`__              |      |           |
+------------------------------------------------------------------+------+-----------+


Datasets CLOSED QA
------------------

1. `starmpcc/Asclepius-Synthetic-Clinical-Notes <https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes?row=61>`__

   1. **Lang**: EN
   2. **Rows**: 20038
   3. **Preprocess**:

      1. Choose task ``Question Answering``.
      2. Choose columns ``note``, ``question`` and ``answer``.
      3. Rename column ``note`` to ``context``.
      4. Rename column ``answer`` to ``target``.
      5. Reset indexes.

2. `lionelchg/dolly_closed_qa <https://huggingface.co/datasets/lionelchg/dolly_closed_qa?row=0>`__

   1. **Lang**: EN
   2. **Rows**: 1773
   3. **Preprocess**:

      1. Choose columns ``instruction``, ``context`` and ``response``.
      2. Rename column ``instruction`` to ``question``.
      3. Rename column ``response`` to ``target``.
      4. Reset indexes.

3. `HuggingFaceH4/no_robots <https://huggingface.co/datasets/HuggingFaceH4/no_robots?row=12>`__

   1. **Lang**: EN
   2. **Rows**: 260
   3. **Preprocess**:

      1. Select ``train_sft`` split.
      2. Choose category ``Closed QA``.
      3. Choose columns ``prompt``, ``messages``.
      4. Convert column ``messages`` to string, using f-string.
      5. Rename column ``prompt`` to ``question``.
      6. Reset indexes.
      7. Process column ``messages`` with raw text into two columns ``context`` and ``answer``.

4. `sberquad <https://huggingface.co/datasets/sberquad>`__

   1. **Lang**: RU
   2. **Rows**: 5040
   3. **Preprocess**:

      1. Select ``validation`` split.
      2. Choose columns ``question``, ``context``, ``answers``.
      3. Rename column ``answers`` to ``target``.
      4. Process column ``target`` with raw text to leave just an answer in this column.

5. `RussianNLP/wikiomnia <https://huggingface.co/datasets/RussianNLP/wikiomnia>`__

   1. **Lang**: RU
   2. **Rows**: 173000
   3. **Preprocess**:

      1. Select ``train`` split and ```wikiomnia_ruGPT3_filtered`` subset.
      2. Drop NaN.
      3. Drop duplicates
      4. Reset indexes.
      5. Choose columns ``question``, ``summary``, ``answer``.
      6. Rename columns ``summary`` to ``context`` and ``answer`` to ``target``.

Inferring batch
---------------

Process of implementing method
:py:meth:`lab_7_llm.main.LLMPipeline._infer_batch`
for closed question-answering task has its specifics:

   1. You need to transpose the ``sample_batch`` before you pass it to the tokenizer,
      so that it is a sequence of tuples
      where each tuple has two strings: a question and a context.
   2. The prediction of the model will consist of two tensors
      that contain start and end scores respectively.
   3. Only the ids between start and end location corresponding
      to the answer have to be decoded and passed on.
   4. To get the ids, iterate through ``input_ids`` field of the tokenized batch.

Metrics CLOSED QA
-----------------

-  squad

.. note:: To calculate the squad metric, you need to convert the data
          into a special structure. This structure you can find in
          `this repository <https://github.com/huggingface/datasets>`__
          in the ``metrics`` directory.

Datasets OPEN QA
----------------

1. `truthful_qa <https://huggingface.co/datasets/truthful_qa>`__

   1. **Lang**: EN
   2. **Rows**: 817
   3. **Preprocess**:

      1. Drop columns ``type``, ``category``, ``correct_answers``,
         ``incorrect_answers``, ``source``.
      2. Rename column ``best_answer`` to ``target``.

2. `jtatman/databricks-dolly-8k-qa-open-close <https://huggingface.co/datasets/jtatman/databricks-dolly-8k-qa-open-close>`__

   1. **Lang**: EN
   2. **Rows**: 7706
   3. **Preprocess**:

      1. Filter dataset rows by ``category`` == ``open_qa``.
      2. Drop columns ``context``, ``category``, ``__index_level_0__``.
      3. Rename column ``instruction`` to ``question``.
      4. Rename column ``response`` to ``target``.

3. `tatsu-lab/alpaca <https://huggingface.co/datasets/tatsu-lab/alpaca>`__

   1. **Lang**: EN
   2. **Rows**: 52002
   3. **Preprocess**:

      1. Drop columns ``input``, ``text``.
      2. Rename column ``instruction`` to ``question``.
      3. Rename column ``output`` to ``target``.

4. `lionelchg/dolly_open_qa <https://huggingface.co/datasets/lionelchg/dolly_open_qa>`__

   1. **Lang**: EN
   2. **Rows**: 188
   3. **Preprocess**:

      1. Drop columns ``context``, ``category``, ``text``.
      2. Rename column ``instruction`` to ``question``.
      3. Rename column ``response`` to ``target``.

Metrics OPEN QA
---------------

-  BLEU
-  ROUGE

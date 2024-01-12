.. _generation-label:

Generation
==========

Models
------

+------------------------------------------------------------------+------+-----------+
| Model                                                            | Lang | Task      |
+==================================================================+======+===========+
| `GPT2 <https://huggingface.co/gpt2>`__                           | EN   | GENERATION|
+------------------------------------------------------------------+------+-----------+
| `rugpt3small                                                     | RU   | CLOSED QA |
| <https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2>`__ |      |           |
+------------------------------------------------------------------+------+-----------+
| `timpal0l/mdeberta-v3-base-squad2 <https://                      | EN   | CLOSED QA |
| <https://huggingface.co/timpal0l/mdeberta-v3-base-squad2>`__     |      |           |
+------------------------------------------------------------------+------+-----------+
| `VMware/electra-small-mrqa                                       | EN   | CLOSED QA |
| <https://huggingface.co/VMware/electra-small-mrqa>`__            |      |           |
+------------------------------------------------------------------+------+-----------+


Datasets
--------

1. `starmpcc/Asclepius-Synthetic-Clinical-Notes <https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes?row=61>`__

   1. **Lang**: EN
   2. **Task**: Closed QA
   3. **Rows**: 20038
   4. **Preprocess**:

      1. Choose task ``Question Answering``.
      2. Choose columns ``note``, ``question`` and ``answer``.
      3. Rename column ``note`` to ``context``.
      4. Reset indexes.

2. `lionelchg/dolly_closed_qa <https://huggingface.co/datasets/lionelchg/dolly_closed_qa?row=0>`__

   1. **Lang**: EN
   2. **Task**: Closed QA
   3. **Rows**: 1773
   4. **Preprocess**:

      1. Choose columns ``instruction``, ``context`` and ``response``.
      2. Rename columns ``instruction`` to ``question`` and ``response`` to ``target``.
      3. Reset indexes.

3. `RussianNLP/wikiomnia wikiomnia_ruGPT3_filtered <https://huggingface.co/datasets/RussianNLP/wikiomnia?row=1>`__

   1. **Lang**: RU
   2. **Task**: Closed QA
   3. **Rows**: 173314
   4. **Preprocess**:

      1. Choose columns ``question``, ``summary`` and ``answer``.
      2. Rename column ``summary`` to ``context``.

4. `HuggingFaceH4/no_robots <https://huggingface.co/datasets/HuggingFaceH4/no_robots?row=12>`__

   1. **Lang**: EN
   2. **Task**: Closed QA

      1. **Rows**: 260

   3. **Task**: Open QA

      1. **Rows**: 1240

   4. **Task**: Generation

      1. **Rows**: 4560

   5. **Preprocess**:

      1. Choose category ``Closed QA``.
      2. Choose columns ``prompt``, ``messages``.
      3. Rename column ``prompt`` to ``question``.
      4. Reset indexes.
      5. Process column ``messages`` with raw text into two columns ``context`` and ``answer``.

Inferring batch
---------------
Process of implementing method
:py:meth:`lab_7_llm.main.LLMPipeline._infer_batch`
for question-answering task has its specifics.

1. You need to transpose the sample_batch before you pass it to the tokenizer,
so that it is a sequence of tuples where each tuple has two strings: a question and a context.
2. The prediction of the model will consist of two tensors
that contain start scores and end score respectively.
3. Only the ids between start and end location corresponding
to the answer have to be decoded and passed on.
4. To get the ids, iterate through ``input_ids`` field of the tokenized batch.

Metrics
-------

-  squad

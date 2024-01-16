.. _nli-label:

NLI
===

Models
------

+-------------------------------------------------------------------+------+
| Model                                                             | Lang |
+===================================================================+======+
| `cointegrated/rubert-base-cased-nli-threeway <https://            | RU   |
| huggingface.co/cointegrated/rubert-base-cased-nli-threeway>`__    |      |
+-------------------------------------------------------------------+------+
| `cointegrated/rubert-tiny-bilingual-nli                           | RU   |
| <face.co/cointegrated/rubert-tiny-bilingual-nli>`__               |      |
+-------------------------------------------------------------------+------+
| `cross-encoder/qnli-distilroberta-base                            | EN   |
| <https://huggingface.co/cross-encoder/qnli-distilroberta-base>`__ |      |
+-------------------------------------------------------------------+------+
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli <https:             | EN   |
| //huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli>`__ |      |
+-------------------------------------------------------------------+------+


Datasets
--------

1. `nli-rus-translated-v2021 <https://huggingface.co/datasets/cointegrated/nli-rus-translated-v2021>`__

   1. **Lang**: RU
   2. **Rows**: 19647
   3. **Preprocess**:

      1. Filter the dataset by the column ``source`` with the value ``mnli``.
            ( This step you should implement in obtain method )
      2. Leave only columns ``premise_ru``, ``hypothesis_ru`` and ``label``.
      3. Rename column ``premise_ru`` to ``premise``.
      4. Rename column ``hypothesis_ru`` to ``hypothesis``.
      5. Rename column ``label`` to  ``target``.
      6. Map ``target`` with class labels.
      7. Delete empty rows in dataset.
      8. Reset index.

2. `Russian Super GLUE TERRA <https://huggingface.co/datasets/RussianNLP/russian_super_glue>`__

   1. **Lang**: RU
   2. **Rows**: 307
   3. **Preprocess**:

      1. Rename column ``label`` to  ``target``.
      2. Delete duplicates in dataset.
      3. Delete empty rows in dataset.
      4. Reset index.

3. `XNLI <https://huggingface.co/datasets/xnli>`__

   1. **Lang**: RU
   2. **Rows**: 2490
   3. **Preprocess**:

      1. Rename column ``label`` to  ``target``.
      2. Delete duplicates in dataset.
      3. Delete empty rows in dataset.
      4. Reset index.

4. `GLUE QNLI <https://huggingface.co/datasets/glue>`__

   1. **Lang**: EN
   2. **Rows**: 5463
   3. **Preprocess**:

      1. Rename column ``question`` to  ``premise``.
      2. Rename column ``sentence`` to  ``hypothesis``.
      3. Rename column ``label`` to  ``target``.
      4. Delete duplicates in dataset.
      5. Delete empty rows in dataset.
      6. Reset index.

5. `GLUE MNLI <https://huggingface.co/datasets/glue>`__

   1. **Lang**: EN
   2. **Rows**: 9815
   3. **Preprocess**:

      1. Rename column ``label`` to  ``target``.
      2. Delete duplicates in dataset.
      3. Delete empty rows in dataset.
      4. Reset index.

Metrics
-------

-  Accuracy

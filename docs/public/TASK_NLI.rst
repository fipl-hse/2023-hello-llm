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
      2. Cut to 100 rows.
      3. Leave only columns ``premise_ru``, ``hypothesis_ru`` and ``label``.
      4. Rename column ``premise_ru`` to ``premise``.
      5. Rename column ``hypothesis_ru`` to ``hypothesis``.
      6. Rename column ``label`` to  ``target``.
      7. Map ``target`` with class labels.
      8. Delete empty rows in dataset.
      9. Reset index.

2. `Russian Super GLUE TERRA <https://huggingface.co/datasets/RussianNLP/russian_super_glue>`__

   1. **Lang**: RU
   2. **Rows**: 307
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Rename column ``label`` to  ``target``.
      3. Delete duplicates in dataset.
      4. Delete empty rows in dataset.
      5. Reset index.

3. `XNLI <https://huggingface.co/datasets/xnli>`__

   1. **Lang**: RU
   2. **Rows**: 2490
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Rename column ``label`` to  ``target``.
      3. Delete duplicates in dataset.
      4. Delete empty rows in dataset.
      5. Reset index.

4. `GLUE QNLI <https://huggingface.co/datasets/glue>`__

   1. **Lang**: EN
   2. **Rows**: 5463
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Rename column ``question`` to  ``premise``.
      3. Rename column ``sentence`` to  ``hypothesis``.
      4. Rename column ``label`` to  ``target``.
      5. Delete duplicates in dataset.
      6. Delete empty rows in dataset.
      7. Reset index.

5. `GLUE MNLI <https://huggingface.co/datasets/glue>`__

   1. **Lang**: EN
   2. **Rows**: 9815
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Rename column ``label`` to  ``target``.
      3. Delete duplicates in dataset.
      4. Delete empty rows in dataset.
      5. Reset index.

Metrics
-------

-  Accuracy

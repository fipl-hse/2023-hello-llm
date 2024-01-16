.. _classification-label:

Classification
==============

Models
------

+---------------------------------------------------------------------+------+
| Model                                                               | Lang |
+=====================================================================+======+
| `cointegrated/rubert-tiny-toxicity <https                           | EN   |
| ://huggingface.co/cointegrated/rubert-tiny-toxicityr>`__            |      |
+---------------------------------------------------------------------+------+
| `cointegrated/rubert-tiny2-cedr-emotion-detection <https://hugging  | RU   |
| face.co/cointegrated/rubert-tiny2-cedr-emotion-detection>`__        |      |
+---------------------------------------------------------------------+------+
| `papluca/xlm-roberta-base-language-detection <https://hugging       | RU   |
| face.co/papluca/xlm-roberta-base-language-detection>`__             |      |
+---------------------------------------------------------------------+------+
| `fabriceyhc/bert-base-uncased-ag_news <https://hugging              | EN   |
| face.co/fabriceyhc/bert-base-uncased-ag_news>`__                    |      |
+---------------------------------------------------------------------+------+
| `XSY/albert-base-v2-imdb-calssification <https://hugging            | EN   |
| face.co/XSY/albert-base-v2-imdb-calssification>`__                  |      |
+---------------------------------------------------------------------+------+
| `aiknowyou/it-emotion-analyzer <https://hugging                     | RU   |
| face.co/aiknowyou/it-emotion-analyzer>`__                           |      |
+---------------------------------------------------------------------+------+

Datasets
--------

1. `OxAISH-AL-LLM/wiki_toxic <https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic/viewer/default/validation>`__

   1. **Lang**: EN
   2. **Rows**: 31915
   3. **Preprocess**:

      1. Drop column ``id``.
      2. Rename column ``labels`` to ``label``.
      3. Rename column ``text`` to ``target``.
      4. Reset indexes

2. `seara/ru_go_emotions <https://huggingface.co/datasets/seara/ru_go_emotions>`__

   1. **Lang**: RU
   2. **Rows**: 5430
   3. **Preprocess**:

      1. Drop columns ``id`` and ``text``.
      2. Rename columns ``labels`` to ``label``.
      3. Rename column ``text`` to ``target``.
      4. Group emotions and change numbers to words.
      5. Delete duplicates in ``label``.
      6. Clean column ``target``.
      7. Reset indexes

3. `papluca/language-identification <https://huggingface.co/datasets/papluca/language-identification>`__

   1. **Lang**: EN
   2. **Rows**: 10000
   3. **Preprocess**:

      1. Rename column ``labels`` to ``label``.
      2. Rename column ``text`` to ``target``.
      3. Map language abbreviation to label classes.
      4. Reset indexes

4. `ag_news <https://huggingface.co/datasets/ag_news>`__

   1. **Lang**: EN
   2. **Rows**: 7600
   3. **Preprocess**:

      1. Rename column ``labels`` to ``label``.
      2. Rename column ``text`` to ``target``.
      3. Reset indexes

5. `imdb <https://huggingface.co/datasets/imdb>`__

   1. **Lang**: EN
   2. **Rows**: 25000
   3. **Preprocess**:

      1. Rename column ``labels`` to ``label``.
      2. Rename column ``text`` to ``target``.
      3. Reset indexes

6. `dair-ai/emotion <https://huggingface.co/datasets/dair-ai/emotion>`__

   1. **Lang**: EN
   2. **Rows**: 2000
   3. **Preprocess**:

      1. Rename column ``labels`` to ``label``.
      2. Rename column ``text`` to ``target``.
      3. Reset indexes

Metrics
-------

-  F1-score

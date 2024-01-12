.. _classification-label:

Classification
==============


.. contents:: Content
   :depth: 2


Emotion detection
-----------------

Models
~~~~~~

+------------------------------------------------------------------------------+------+
| Model                                                                        | Lang |
+==============================================================================+======+
| `DistilRoBERTa-base                                                          | EN   |
| <https://huggingface.co/michellejieli/emotion_text_classifier>`__            |      |
+------------------------------------------------------------------------------+------+
| `rubert-tiny2                                                                | RU   |
| <https://huggingface.co/cointegrated/rubert-tiny2-cedr-emotion-detection>`__ |      |
+------------------------------------------------------------------------------+------+

Datasets
~~~~~~~~

1. `go_emotions <https://huggingface.co/datasets/go_emotions>`__

   1. **Lang**: EN
   2. **Rows**: 5430
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Drop column ``id``.
      3. Rename column ``labels`` to ``target``.
      4. Remove unnecessary emotions
         ``[0, 4, 5, 6, 7, 8, 10, 12, 15, 18, 21, 22, 23]``.
      5. Group emotions:

         1. joy - ``[1, 13, 17, 20]``
         2. sadness - ``[9, 16, 24, 25]``
         3. fear - ``[14, 19]``
         4. anger - ``[2, 3]``
         5. disgust - ``11``
         6. surprise - ``26``
         7. neutral - ``27``
         8. other

      6. Delete duplicates in ``label``.
      7. Clean column ``text``.

2. `emotion-detection-from-text <https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text>`__

   1. **Lang**: EN
   2. **Rows**: 39827
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Drop column ``tweet_id``.
      3. Rename columns ``sentiment`` to ``target`` and ``content`` to ``source``.
      4. Remove unnecessary emotions ``['love', 'boredom', 'relief', 'worry']``.
      5. Group emotions:

         1. joy - ``['enthusiasm', 'fun', 'happiness']``
         2. sadness - ``empty``
         3. anger - ``hate``

      6. Clean column ``text``.

3. `seara/ru_go_emotions <https://huggingface.co/datasets/seara/ru_go_emotions>`__

   1. **Lang**: RU
   2. **Rows**: 5430
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Drop columns ``id`` and ``text``.
      3. Rename columns ``labels`` to ``target`` and ``ru_text`` to
         ``source``.
      4. Remove unnecessary emotions
         ``[0, 4, 5, 6, 7, 8, 10, 11, 12, 15, 18, 21, 22, 23]``.
      5. Group emotions:

         1. joy - ``[1, 13, 17, 20]``
         2. sadness - ``[9, 16, 24, 25]``
         3. fear - ``[14, 19]``
         4. anger - ``[2, 3]``
         5. surprise - ``26``
         6. neutral - ``27``
         7. other

      6. Delete duplicates in ``label``.
      7. Clean column ``text``.

Toxicity detection
------------------

Models
~~~~~~

+----------------------------------------------------------------------------+------+
| Model                                                                      | Lang |
+============================================================================+======+
| `roberta <https://huggingface.co/cointegrated/rubert-tiny-toxicity>`__     | EN   |
+----------------------------------------------------------------------------+------+
| `DistilBERT <https://huggingface.co/martin-ha/toxic-comment-model>`__      | EN   |
+----------------------------------------------------------------------------+------+
| `rubert-tiny <https://huggingface.co/cointegrated/rubert-tiny-toxicity>`__ | RU   |
+----------------------------------------------------------------------------+------+


Datasets
~~~~~~~~

1. `OxAISH-AL-LLM/wiki_toxic <https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic>`__

   1. **Lang**: EN
   2. **Rows**: 25900
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Drop column ``id``.
      3. Rename column ``comment_text`` to ``source``.
      4. Clean column ``text``.

2. `toxic-tweets-dataset <https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset>`__

   1. **Lang**: EN
   2. **Rows**: 54313
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Drop column ``Unnamed: 0``.
      3. Rename columns ``Toxicity`` to ``target`` and ``tweet`` to ``source``.
      4. Clean column ``text``.

3. `russian-language-toxic-comments <https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments>`__

   1. **Lang**: RU
   2. **Rows**: 14412
   3. **Preprocess**:

      1. Cut to 100 rows.
      2. Rename columns ``toxic`` to ``target`` and ``comment`` to ``source``.
      3. Clean column ``text``.

Metrics
-------

-  Precision
-  Recall
-  F1-score

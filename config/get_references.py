"""
Collect and store model analytics.
"""
# pylint: disable=import-error, too-many-branches, no-else-return, inconsistent-return-statements, wrong-import-order
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass

from config.get_model_analytics import get_references, save_reference
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter

from reference_lab_classification.main import (AgNewsDataImporter, AgNewsPreprocessor,  # isort:skip
                                               ClassificationLLMPipeline, DairAiEmotionDataImporter,
                                               DairAiEmotionPreprocessor, GoEmotionsDataImporter,
                                               GoEmotionsRawDataPreprocessor, ImdbDataImporter,
                                               ImdbDataPreprocessor,
                                               LanguageIdentificationDataImporter,
                                               LanguageIdentificationPreprocessor,
                                               RuGoEmotionsRawDataPreprocessor, RuGoRawDataImporter,
                                               WikiToxicDataImporter, WikiToxicRawDataPreprocessor)
from reference_lab_generation.main import (ClinicalNotesRawDataImporter,  # isort:skip
                                           ClinicalNotesRawDataPreprocessor,
                                           DollyClosedRawDataImporter,
                                           DollyClosedRawDataPreprocessor, GenerationLLMPipeline,
                                           GenerationTaskDataset, GenerationTaskEvaluator,
                                           NoRobotsRawDataImporter, NoRobotsRawDataPreprocessor)
from reference_lab_nli.main import (DatasetTypes, GlueDataImporter,  # isort:skip
                                    NliDataPreprocessor,
                                    NLILLMPipeline, NliRusDataImporter,
                                    NliRusTranslatedDataPreprocessor, NliTaskDataset,
                                    QnliDataPreprocessor, RussianSuperGlueDataImporte,
                                    XnliDataImporter)
from reference_lab_nmt.helpers import (EnDeRawDataPreprocessor, RuEnRawDataImporter,  # isort:skip
                                       RuEnRawDataPreprocessor, RuEsRawDataPreprocessor)
from reference_lab_nmt.main import (LLMPipeline,  # isort:skip
                                    RawDataImporter,
                                    RawDataPreprocessor, TaskDataset, TaskEvaluator)
from reference_lab_summarization.main import (DailymailRawDataImporter,  # isort:skip
                                              DailymailRawDataPreprocessor,
                                              GovReportRawDataPreprocessor,
                                              PubMedRawDataPreprocessor, RuCorpusRawDataImporter,
                                              RuCorpusRawDataPreprocessor,
                                              RuDialogNewsRawDataPreprocessor,
                                              RuGazetaRawDataPreprocessor, RuReviewsRawDataImporter,
                                              RuReviewsRawDataPreprocessor,
                                              ScientificLiteratureRawDataImporter,
                                              ScientificLiteratureRawDataPreprocessor,
                                              SummarizationRawDataImporter)


@dataclass
class MainParams:
    """
    Main parameters.
    """
    model: str
    dataset: str
    metrics: list[Metrics]


@dataclass
class InferenceParams:
    """
    Inference parameters.
    """
    num_samples: int
    max_length: int
    batch_size: int
    predictions_path: Path
    device: str


def nmt_inference(main_params: MainParams,
                  inference_params: InferenceParams) -> Any:
    """
    Gets inference for nmt task.

    Args:
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric value
    """
    if main_params.dataset == 'shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2':
        importer = RuEnRawDataImporter(main_params.dataset)
    else:
        importer = RawDataImporter(main_params.dataset)
    importer.obtain()
    if importer.raw_data is None:
        raise ValueError('Unable to process data which is None!')

    if main_params.dataset == 'shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2':
        preprocessor = RuEnRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl':
        preprocessor = RuEsRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'RocioUrquijo/en_de':
        preprocessor = EnDeRawDataPreprocessor(importer.raw_data)
    else:
        preprocessor = RawDataPreprocessor(importer.raw_data)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(inference_params.num_samples))
    pipeline = LLMPipeline(main_params.model,
                           dataset,
                           inference_params.max_length,
                           inference_params.batch_size,
                           inference_params.device)

    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(inference_params.predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(data_path=inference_params.predictions_path,
                              metrics=main_params.metrics)
    result = evaluator.run()
    return result


def generation_inference(main_params: MainParams,
                         inference_params: InferenceParams) -> Any:
    """
    Gets inference for generation task.

    Args:
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric value
    """
    # START OF LEVEL 4
    importer: AbstractRawDataImporter
    if main_params.dataset == 'lionelchg/dolly_closed_qa':
        importer = DollyClosedRawDataImporter(main_params.dataset)
    elif main_params.dataset == 'starmpcc/Asclepius-Synthetic-Clinical-Notes':
        importer = ClinicalNotesRawDataImporter(main_params.dataset)
    else:
        importer = NoRobotsRawDataImporter(main_params.dataset)
    importer.obtain()
    if importer.raw_data is None:
        raise ValueError('Unable to process data which is None!')

    if main_params.dataset == 'lionelchg/dolly_closed_qa':
        preprocessor = DollyClosedRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'starmpcc/Asclepius-Synthetic-Clinical-Notes':
        preprocessor = ClinicalNotesRawDataPreprocessor(importer.raw_data)
    else:
        preprocessor = NoRobotsRawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    if preprocessor.data is None:
        raise ValueError('Data can not be empty!')

    dataset = GenerationTaskDataset(preprocessor.data.head(inference_params.num_samples))

    pipeline = GenerationLLMPipeline(main_params.model,
                                     dataset,
                                     inference_params.max_length,
                                     inference_params.batch_size,
                                     inference_params.device)
    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(inference_params.predictions_path, index=False, encoding='utf-8')

    evaluator = GenerationTaskEvaluator(
        data_path=inference_params.predictions_path,
        metrics=main_params.metrics
    )
    result = evaluator.run()
    return result


def classification_inference(main_params: MainParams,
                             inference_params: InferenceParams) -> Any:
    """
    Gets inference for classification task.

    Args:
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric value
    """
    importer: AbstractRawDataImporter
    if main_params.dataset == 'seara/ru_go_emotions':
        importer = RuGoRawDataImporter(main_params.dataset)
    elif main_params.dataset == 'imdb':
        importer = ImdbDataImporter(main_params.dataset)
    elif main_params.dataset == 'dair-ai/emotion':
        importer = DairAiEmotionDataImporter(main_params.dataset)
    elif main_params.dataset == 'ag_news':
        importer = AgNewsDataImporter(main_params.dataset)
    elif main_params.dataset == 'papluca/language-identification':
        importer = LanguageIdentificationDataImporter(
            main_params.dataset)
    elif main_params.dataset == 'OxAISH-AL-LLM/wiki_toxic':
        importer = WikiToxicDataImporter(main_params.dataset)
    else:
        importer = GoEmotionsDataImporter(main_params.dataset)
    importer.obtain()
    if importer.raw_data is None:
        raise ValueError('Unable to process data which is None!')

    if main_params.dataset == 'OxAISH-AL-LLM/wiki_toxic':
        preprocessor = WikiToxicRawDataPreprocessor(
            importer.raw_data)
    elif main_params.dataset == 'go_emotions':
        preprocessor = GoEmotionsRawDataPreprocessor(
            importer.raw_data)
    elif main_params.dataset == 'seara/ru_go_emotions':
        preprocessor = RuGoEmotionsRawDataPreprocessor(
            importer.raw_data)
    elif main_params.dataset == 'imdb':
        preprocessor = ImdbDataPreprocessor(
            importer.raw_data)
    elif main_params.dataset == 'dair-ai/emotion':
        preprocessor = DairAiEmotionPreprocessor(
            importer.raw_data)
    elif main_params.dataset == 'ag_news':
        preprocessor = AgNewsPreprocessor(importer.raw_data)
    else:
        preprocessor = LanguageIdentificationPreprocessor(
            importer.raw_data)

    # START OF LEVEL 6
    preprocessor.transform()
    if preprocessor.data is None:
        raise ValueError('Data can not be empty!')

    dataset = TaskDataset(preprocessor.data.head(inference_params.num_samples))

    pipeline = ClassificationLLMPipeline(main_params.model,
                                         dataset,
                                         inference_params.max_length,
                                         inference_params.batch_size,
                                         inference_params.device)

    # START OF LEVEL 8
    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(inference_params.predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(
        data_path=inference_params.predictions_path, metrics=main_params.metrics)
    result = evaluator.run()
    return result


def nli_inference(main_params: MainParams,
                  inference_params: InferenceParams) -> Any:
    """
    Gets inference for nli task.

    Args:
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric value
    """
    dataset_type = DatasetTypes(main_params.dataset)

    importer: AbstractRawDataImporter
    if dataset_type == DatasetTypes.XNLI:
        importer = XnliDataImporter(main_params.dataset)
        importer.obtain()
    if dataset_type == DatasetTypes.NLI_RUS:
        importer = NliRusDataImporter(main_params.dataset)
        importer.obtain()
    if dataset_type == DatasetTypes.MNLI:
        importer = GlueDataImporter(main_params.dataset)
        importer.obtain()
    if dataset_type == DatasetTypes.QNLI:
        importer = GlueDataImporter(main_params.dataset)
        importer.obtain()
    if dataset_type == DatasetTypes.TERRA:
        importer = RussianSuperGlueDataImporte(main_params.dataset)
        importer.obtain()
    if importer.raw_data is None:
        raise ValueError('Unable to process data which is None!')

    if dataset_type not in (DatasetTypes.NLI_RUS, DatasetTypes.QNLI):
        preprocessor = NliDataPreprocessor(importer.raw_data)
    if dataset_type == DatasetTypes.NLI_RUS:
        preprocessor = NliRusTranslatedDataPreprocessor(
            importer.raw_data)
    if dataset_type == DatasetTypes.QNLI:
        preprocessor = QnliDataPreprocessor(importer.raw_data)

    # START OF LEVEL 6
    preprocessor.transform()
    if preprocessor.data is None:
        raise ValueError('Data can not be empty!')
    dataset = NliTaskDataset(preprocessor.data.head(inference_params.num_samples))

    pipeline = NLILLMPipeline(main_params.model,
                              dataset,
                              inference_params.max_length,
                              inference_params.batch_size,
                              inference_params.device)

    # START OF LEVEL 8
    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(inference_params.predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(data_path=inference_params.predictions_path,
                              metrics=main_params.metrics)
    result = evaluator.run()
    return result


def summarization_inference(main_params: MainParams,
                            inference_params: InferenceParams) -> Any:
    """
    Gets inference for summarization task.

    Args:
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric value
    """
    # START OF LEVEL 4
    importer: AbstractRawDataImporter
    if main_params.dataset == 'tomasg25/scientific_lay_summarisation':
        importer = ScientificLiteratureRawDataImporter(main_params.dataset)
    elif main_params.dataset == 'cnn_dailymail':
        importer = DailymailRawDataImporter(main_params.dataset)
    elif main_params.dataset == 'd0rj/curation-corpus-ru':
        importer = RuCorpusRawDataImporter(main_params.dataset)
    elif main_params.dataset == 'trixdade/reviews_russian':
        importer = RuReviewsRawDataImporter(main_params.dataset)
    else:
        importer = SummarizationRawDataImporter(main_params.dataset)
    importer.obtain()
    if importer.raw_data is None:
        raise ValueError('Unable to process data which is None!')

    if main_params.dataset == 'ccdv/pubmed-summarization':
        preprocessor = PubMedRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'tomasg25/scientific_lay_summarisation':
        preprocessor = ScientificLiteratureRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'ccdv/govreport-summarization':
        preprocessor = GovReportRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'cnn_dailymail':
        preprocessor = DailymailRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'IlyaGusev/gazeta':
        preprocessor = RuGazetaRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'CarlBrendt/Summ_Dialog_News':
        preprocessor = RuDialogNewsRawDataPreprocessor(importer.raw_data)
    elif main_params.dataset == 'trixdade/reviews_russian':
        preprocessor = RuReviewsRawDataPreprocessor(importer.raw_data)
    else:
        preprocessor = RuCorpusRawDataPreprocessor(importer.raw_data)

    _ = preprocessor.analyze()
    # END OF LEVEL 4

    # START OF LEVEL 6
    preprocessor.transform()
    if preprocessor.data is None:
        raise ValueError('Data can not be empty!')
    dataset = TaskDataset(preprocessor.data.head(inference_params.num_samples))

    pipeline = LLMPipeline(main_params.model,
                           dataset,
                           inference_params.max_length,
                           inference_params.batch_size,
                           inference_params.device)

    # START OF LEVEL 8
    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(inference_params.predictions_path, index=False, encoding='utf-8')

    evaluator = TaskEvaluator(data_path=inference_params.predictions_path,
                              metrics=main_params.metrics)
    result = evaluator.run()
    return result
    # END OF LEVEL 8


def get_task(model: str, main_params: MainParams, inference_params: InferenceParams) -> Any:
    """
    Gets task.

    Args:
        model (str): name of model
        main_params (MainParams): Parameters from main
        inference_params (InferenceParams): Parameters from inference

    Returns:
        Any: Metric for a specific task
    """
    nmt_model = [
        'Helsinki-NLP/opus-mt-en-fr',
        'Helsinki-NLP/opus-mt-ru-en',
        'Helsinki-NLP/opus-mt-ru-es',
        't5-small'
    ]

    generation_model = [
        'VMware/electra-small-mrqa',
        'timpal0l/mdeberta-v3-base-squad2'
    ]

    classification_model = [
        'cointegrated/rubert-tiny-toxicity',
        'cointegrated/rubert-tiny2-cedr-emotion-detection',
        'papluca/xlm-roberta-base-language-detection',
        'fabriceyhc/bert-base-uncased-ag_news',
        'XSY/albert-base-v2-imdb-calssification',
        'aiknowyou/it-emotion-analyzer'
    ]

    nli_model = [
        'cointegrated/rubert-base-cased-nli-threeway',
        'cointegrated/rubert-tiny-bilingual-nli',
        'cross-encoder/qnli-distilroberta-base',
        'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    ]

    summarization_model = [
        'mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization',
        'nandakishormpai/t5-small-machine-articles-tag-generation',
        'mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization',
        'stevhliu/my_awesome_billsum_model',
        'UrukHan/t5-russian-summarization',
        'dmitry-vorobiev/rubert_ria_headlines'
    ]

    if model in nmt_model:
        return nmt_inference(main_params, inference_params)
    elif model in generation_model:
        return generation_inference(main_params, inference_params)
    elif model in classification_model:
        return classification_inference(main_params, inference_params)
    elif model in nli_model:
        return nli_inference(main_params, inference_params)
    elif model in summarization_model:
        return summarization_inference(main_params, inference_params)


def main() -> None:
    """
    Run collected reference scores.
    """
    references_path = Path(__file__).parent / 'reference_scores.json'
    dest = Path(__file__).parent / 'reference_score_.json'

    max_length = 120
    batch_size = 1
    num_samples = 100

    inference_params = InferenceParams(num_samples,
                                       max_length,
                                       batch_size,
                                       Path('result.csv'),
                                       'cpu')

    references = get_references(path=references_path)
    result = {}
    for model, datasets in references.items():
        result[model] = {}
        for dataset, metrics in datasets.items():
            result[model][dataset] = {}
            for metric in metrics:
                result[model][dataset][metric] = {}
                if 'test_' in model:
                    continue

                print(model, dataset, metric)
                main_params = MainParams(model, dataset, [Metrics(metric)])
                inference_func = get_task(model, main_params, inference_params)
                metric = f'{inference_func[metric]:.5f}'
                result[model][dataset][metric] = metric

    save_reference(dest, result)


if __name__ == '__main__':
    main()

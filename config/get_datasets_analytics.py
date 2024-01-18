"""
Collects and stores dataset analytics
"""
# pylint: disable=R0912,R0915
# mypy: ignore-errors
from pathlib import Path

from tqdm import tqdm

from config.get_model_analytics import get_references, save_reference
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from reference_lab_classification.main import (AgNewsDataImporter, AgNewsPreprocessor,
                                               DairAiEmotionDataImporter, DairAiEmotionPreprocessor,
                                               GoEmotionsDataImporter,
                                               GoEmotionsRawDataPreprocessor, ImdbDataImporter,
                                               ImdbDataPreprocessor,
                                               LanguageIdentificationDataImporter,
                                               LanguageIdentificationPreprocessor,
                                               RuGoEmotionsRawDataPreprocessor, RuGoRawDataImporter,
                                               WikiToxicDataImporter, WikiToxicRawDataPreprocessor)
from reference_lab_generation.main import (ClinicalNotesRawDataImporter,
                                           ClinicalNotesRawDataPreprocessor,
                                           DollyClosedRawDataImporter,
                                           DollyClosedRawDataPreprocessor, NoRobotsRawDataImporter,
                                           NoRobotsRawDataPreprocessor)
from reference_lab_nli.main import (DatasetTypes, GlueDataImporter, NliDataPreprocessor,
                                    NliRusDataImporter, NliRusTranslatedDataPreprocessor,
                                    QnliDataPreprocessor, RussianSuperGlueDataImporte,
                                    XnliDataImporter)
from reference_lab_nmt.helpers import (EnDeRawDataPreprocessor, RuEnRawDataImporter,
                                       RuEnRawDataPreprocessor, RuEsRawDataPreprocessor)
from reference_lab_nmt.main import RawDataImporter, RawDataPreprocessor
from reference_lab_summarization.main import (DailymailRawDataImporter,
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


def main() -> None:
    """
    Run the collect dataset analytics.
    """

    references_path = Path(__file__).parent / 'reference_scores.json'
    dest = Path(__file__).parent / 'reference_dataset_analytics.json'

    references = get_references(path=references_path)

    datasets_to_analyze = []
    for model, dataset_pack in references.items():
        for dataset_name in dataset_pack.keys():
            datasets_to_analyze.append(dataset_name)

    datasets_to_analyze = set(datasets_to_analyze)

    result = {}
    for dataset_name in tqdm(datasets_to_analyze):
        importer: AbstractRawDataImporter
        if dataset_name == 'seara/ru_go_emotions':
            importer = RuGoRawDataImporter(dataset_name)
        elif dataset_name == 'imdb':
            importer = ImdbDataImporter(dataset_name)
        elif dataset_name == 'dair-ai/emotion':
            importer = DairAiEmotionDataImporter(dataset_name)
        elif dataset_name == 'ag_news':
            importer = AgNewsDataImporter(dataset_name)
        elif dataset_name == 'papluca/language-identification':
            importer = LanguageIdentificationDataImporter(dataset_name)
        elif dataset_name == 'OxAISH-AL-LLM/wiki_toxic':
            importer = WikiToxicDataImporter(dataset_name)
        elif dataset_name == 'go_emotions':
            importer = GoEmotionsDataImporter(dataset_name)
        elif dataset_name == 'lionelchg/dolly_closed_qa':
            importer = DollyClosedRawDataImporter(dataset_name)
        elif dataset_name == 'starmpcc/Asclepius-Synthetic-Clinical-Notes':
            importer = ClinicalNotesRawDataImporter(dataset_name)
        elif dataset_name == 'HuggingFaceH4/no_robots':
            importer = NoRobotsRawDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.XNLI.value:
            importer = XnliDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.NLI_RUS.value:
            importer = NliRusDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.MNLI.value:
            importer = GlueDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.QNLI.value:
            importer = GlueDataImporter(dataset_name)
        elif dataset_name == DatasetTypes.TERRA.value:
            importer = RussianSuperGlueDataImporte(dataset_name)
        elif dataset_name == 'tomasg25/scientific_lay_summarisation':
            importer = ScientificLiteratureRawDataImporter(dataset_name)
        elif dataset_name == 'cnn_dailymail':
            importer = DailymailRawDataImporter(dataset_name)
        elif dataset_name == 'd0rj/curation-corpus-ru':
            importer = RuCorpusRawDataImporter(dataset_name)
        elif dataset_name == 'trixdade/reviews_russian':
            importer = RuReviewsRawDataImporter(dataset_name)
        elif dataset_name in ['ccdv/pubmed-summarization',
                              'ccdv/govreport-summarization',
                              'IlyaGusev/gazeta',
                              'CarlBrendt/Summ_Dialog_News']:
            importer = SummarizationRawDataImporter(dataset_name)
        elif dataset_name == 'shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2':
            importer = RuEnRawDataImporter(dataset_name)
        else:
            importer = RawDataImporter(dataset_name)
        importer.obtain()

        preprocessor: AbstractRawDataPreprocessor
        if dataset_name == 'OxAISH-AL-LLM/wiki_toxic':
            preprocessor = WikiToxicRawDataPreprocessor(
                importer.raw_data)
        elif dataset_name == 'go_emotions':
            preprocessor = GoEmotionsRawDataPreprocessor(
                importer.raw_data)
        elif dataset_name == 'seara/ru_go_emotions':
            preprocessor = RuGoEmotionsRawDataPreprocessor(
                importer.raw_data)
        elif dataset_name == 'imdb':
            preprocessor = ImdbDataPreprocessor(
                importer.raw_data)
        elif dataset_name == 'dair-ai/emotion':
            preprocessor = DairAiEmotionPreprocessor(
                importer.raw_data)
        elif dataset_name == 'ag_news':
            preprocessor = AgNewsPreprocessor(importer.raw_data)
        elif dataset_name == 'papluca/language-identification':
            preprocessor = LanguageIdentificationPreprocessor(
                importer.raw_data)
        elif dataset_name == 'lionelchg/dolly_closed_qa':
            preprocessor = DollyClosedRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'starmpcc/Asclepius-Synthetic-Clinical-Notes':
            preprocessor = ClinicalNotesRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'HuggingFaceH4/no_robots':
            preprocessor = NoRobotsRawDataPreprocessor(importer.raw_data)
        elif dataset_name in (DatasetTypes.XNLI.value,
                              DatasetTypes.MNLI.value,
                              DatasetTypes.TERRA.value):
            preprocessor = NliDataPreprocessor(importer.raw_data)
        elif dataset_name == DatasetTypes.NLI_RUS.value:
            preprocessor = NliRusTranslatedDataPreprocessor(
                importer.raw_data)
        elif dataset_name == DatasetTypes.QNLI.value:
            preprocessor = QnliDataPreprocessor(importer.raw_data)
        elif dataset_name == 'ccdv/pubmed-summarization':
            preprocessor = PubMedRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'tomasg25/scientific_lay_summarisation':
            preprocessor = ScientificLiteratureRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'ccdv/govreport-summarization':
            preprocessor = GovReportRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'cnn_dailymail':
            preprocessor = DailymailRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'IlyaGusev/gazeta':
            preprocessor = RuGazetaRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'CarlBrendt/Summ_Dialog_News':
            preprocessor = RuDialogNewsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'trixdade/reviews_russian':
            preprocessor = RuReviewsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'd0rj/curation-corpus-ru':
            preprocessor = RuCorpusRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2':
            preprocessor = RuEnRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl':
            preprocessor = RuEsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'RocioUrquijo/en_de':
            preprocessor = EnDeRawDataPreprocessor(importer.raw_data)
        else:
            preprocessor = RawDataPreprocessor(importer.raw_data)

        dataset_analysis = preprocessor.analyze()
        result[dataset_name] = dataset_analysis

    save_reference(dest, result)


if __name__ == '__main__':
    main()

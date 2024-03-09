"""
Collect and store dataset analytics.
"""
# pylint: disable=import-error, too-many-branches, too-many-statements, wrong-import-order
import sys
from pathlib import Path

from tqdm import tqdm

from config.get_model_analytics import get_references, save_reference
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor

from reference_lab_classification.main import (AgNewsDataImporter, AgNewsPreprocessor,  # isort:skip
                                               CyrillicTurkicDataImporter,
                                               CyrillicTurkicPreprocessor,
                                               DairAiEmotionDataImporter, DairAiEmotionPreprocessor,
                                               GoEmotionsDataImporter,
                                               GoEmotionsRawDataPreprocessor,
                                               HealthcareDataImporter, HealthcarePreprocessor,
                                               ImdbDataImporter, ImdbDataPreprocessor,
                                               KinopoiskDataImporter, KinopoiskPreprocessor,
                                               LanguageIdentificationDataImporter,
                                               LanguageIdentificationPreprocessor,
                                               RuDetoxifierDataImporter, RuDetoxifierPreprocessor,
                                               RuGoEmotionsRawDataPreprocessor, RuGoRawDataImporter,
                                               RuNonDetoxifiedDataImporter,
                                               RuNonDetoxifiedPreprocessor, RuParadetoxDataImporter,
                                               RuParadetoxPreprocessor, WikiToxicDataImporter,
                                               WikiToxicRawDataPreprocessor,
                                               ToxicityDataPreprocessor,
                                               ParadetoxDataPreprocessor, ToxicityDataImporter,
                                               ParadetoxDataImporter)
from reference_lab_generation.main import (ClinicalNotesRawDataImporter,  # isort:skip
                                           ClinicalNotesRawDataPreprocessor,
                                           DollyClosedRawDataImporter,
                                           DollyClosedRawDataPreprocessor, NoRobotsRawDataImporter,
                                           NoRobotsRawDataPreprocessor, SberquadRawDataImporter,
                                           WikiOmniaRawDataImporter, SberquadRawDataPreprocessor,
                                           WikiOmniaRawDataPreprocessor)
from reference_lab_nli.main import (DatasetTypes, GlueDataImporter,  # isort:skip
                                    NliDataPreprocessor,
                                    NliRusDataImporter, NliRusTranslatedDataPreprocessor,
                                    QnliDataPreprocessor, RussianSuperGlueDataImporte,
                                    XnliDataImporter)
from reference_lab_nmt.main import (RuEnRawDataImporter, RuEnRawDataPreprocessor,  # isort:skip
                                    RuEsRawDataPreprocessor, EnDeRawDataPreprocessor)
from reference_lab_open_qa.main import (AlpacaRawDataPreprocessor,  # isort:skip
                                        DatabricksRawDataPreprocessor,
                                        DollyOpenQARawDataImporter, DollyOpenQARawDataPreprocessor,
                                        QARawDataImporter, TruthfulQARawDataImporter,
                                        TruthfulQARawDataPreprocessor)
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


def main() -> None:
    """
    Run the collect dataset analytics.
    """
    references_path = Path(__file__).parent / 'reference_scores.json'
    dest = Path(__file__).parent / 'reference_dataset_analytics.json'

    references = get_references(path=references_path)

    datasets_raw = []
    for model, dataset_pack in references.items():
        for dataset_name in dataset_pack.keys():
            datasets_raw.append(dataset_name)

    datasets_to_analyze = set(datasets_raw)

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
        elif dataset_name == 'sberquad':
            importer = SberquadRawDataImporter(dataset_name)
        elif dataset_name == 'RussianNLP/wikiomnia':
            importer = WikiOmniaRawDataImporter(dataset_name)
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
        elif dataset_name == 'blinoff/kinopoisk':
            importer = KinopoiskDataImporter(dataset_name)
        elif dataset_name == 'blinoff/healthcare_facilities_reviews':
            importer = HealthcareDataImporter(dataset_name)
        elif dataset_name == 'tatiana-merz/cyrillic_turkic_langs':
            importer = CyrillicTurkicDataImporter(dataset_name)
        elif dataset_name == 's-nlp/ru_paradetox_toxicity':
            importer = RuParadetoxDataImporter(dataset_name)
        elif dataset_name == 's-nlp/ru_non_detoxified':
            importer = RuNonDetoxifiedDataImporter(dataset_name)
        elif dataset_name == 'd0rj/rudetoxifier_data':
            importer = RuDetoxifierDataImporter(dataset_name)
        elif dataset_name == 'truthful_qa':
            importer = TruthfulQARawDataImporter(dataset_name)
        elif dataset_name in ['tatsu-lab/alpaca',
                              'jtatman/databricks-dolly-8k-qa-open-close']:
            importer = QARawDataImporter(dataset_name)
        elif dataset_name == 'lionelchg/dolly_open_qa':
            importer = DollyOpenQARawDataImporter(dataset_name)
        elif dataset_name == 'Arsive/toxicity_classification_jigsaw':
            importer = ToxicityDataImporter(dataset_name)
        elif dataset_name == 's-nlp/en_paradetox_toxicity':
            importer = ParadetoxDataImporter(dataset_name)
        else:
            importer = RawDataImporter(dataset_name)

        importer.obtain()

        if importer.raw_data is None:
            print('Raw data is empty')
            sys.exit(1)
        preprocessor: AbstractRawDataPreprocessor
        if dataset_name == 'OxAISH-AL-LLM/wiki_toxic':
            preprocessor = WikiToxicRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'go_emotions':
            preprocessor = GoEmotionsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'seara/ru_go_emotions':
            preprocessor = RuGoEmotionsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'imdb':
            preprocessor = ImdbDataPreprocessor(importer.raw_data)
        elif dataset_name == 'dair-ai/emotion':
            preprocessor = DairAiEmotionPreprocessor(importer.raw_data)
        elif dataset_name == 'ag_news':
            preprocessor = AgNewsPreprocessor(importer.raw_data)
        elif dataset_name == 'papluca/language-identification':
            preprocessor = LanguageIdentificationPreprocessor(importer.raw_data)
        elif dataset_name == 'lionelchg/dolly_closed_qa':
            preprocessor = DollyClosedRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'starmpcc/Asclepius-Synthetic-Clinical-Notes':
            preprocessor = ClinicalNotesRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'HuggingFaceH4/no_robots':
            preprocessor = NoRobotsRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'sberquad':
            preprocessor = SberquadRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'RussianNLP/wikiomnia':
            preprocessor = WikiOmniaRawDataPreprocessor(importer.raw_data)
        elif dataset_name in (DatasetTypes.XNLI.value,
                              DatasetTypes.MNLI.value,
                              DatasetTypes.TERRA.value):
            preprocessor = NliDataPreprocessor(importer.raw_data)
        elif dataset_name == DatasetTypes.NLI_RUS.value:
            preprocessor = NliRusTranslatedDataPreprocessor(importer.raw_data)
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
        elif dataset_name == 'blinoff/kinopoisk':
            preprocessor = KinopoiskPreprocessor(importer.raw_data)
        elif dataset_name == 'blinoff/healthcare_facilities_reviews':
            preprocessor = HealthcarePreprocessor(importer.raw_data)
        elif dataset_name == 'tatiana-merz/cyrillic_turkic_langs':
            preprocessor = CyrillicTurkicPreprocessor(importer.raw_data)
        elif dataset_name == 's-nlp/ru_paradetox_toxicity':
            preprocessor = RuParadetoxPreprocessor(importer.raw_data)
        elif dataset_name == 's-nlp/ru_non_detoxified':
            preprocessor = RuNonDetoxifiedPreprocessor(importer.raw_data)
        elif dataset_name == 'd0rj/rudetoxifier_data':
            preprocessor = RuDetoxifierPreprocessor(importer.raw_data)
        elif dataset_name == 'truthful_qa':
            preprocessor = TruthfulQARawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'jtatman/databricks-dolly-8k-qa-open-close':
            preprocessor = DatabricksRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'tatsu-lab/alpaca':
            preprocessor = AlpacaRawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'lionelchg/dolly_open_qa':
            preprocessor = DollyOpenQARawDataPreprocessor(importer.raw_data)
        elif dataset_name == 'Arsive/toxicity_classification_jigsaw':
            preprocessor = ToxicityDataPreprocessor(importer.raw_data)
        elif dataset_name == 's-nlp/en_paradetox_toxicity':
            preprocessor = ParadetoxDataPreprocessor(importer.raw_data)
        else:
            preprocessor = RawDataPreprocessor(importer.raw_data)
        try:
            dataset_analysis = preprocessor.analyze()
        except Exception as e:
            print(f'{dataset_name} analysis has some problems!')
            raise e
        result[dataset_name] = dataset_analysis

    save_reference(dest, result)


if __name__ == '__main__':
    main()

"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    raw_data_importer = RawDataImporter("nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl")
    raw_data_importer.obtain()

    raw_data_preprocessor = RawDataPreprocessor(raw_data_importer.raw_data)
    print(raw_data_preprocessor.analyze())

    # result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()

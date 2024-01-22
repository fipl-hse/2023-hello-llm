"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter


@report_time
def main():
    """
    Run the translation pipeline.
    """
    # result = None
    # assert result is not None, "Demo does not work correctly"
    load_set = RawDataImporter('d0rj/curation-corpus-ru')

    return load_set


if __name__ == "__main__":
    main()

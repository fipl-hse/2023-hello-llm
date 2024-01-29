"""
Neural machine translation starter.
"""
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time
from main import RawDataImporter

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    dataset = RawDataImporter('self._hf_data')
    dataset.obtain()



if __name__ == "__main__":
    main()

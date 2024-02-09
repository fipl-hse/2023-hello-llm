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
    data = RawDataImporter
    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()

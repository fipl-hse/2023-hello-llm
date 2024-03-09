"""
Neural machine translation starter.
"""
import json

from config.constants import PROJECT_ROOT
# pylint: disable= too-many-locals
from core_utils.llm.time_decorator import report_time


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(PROJECT_ROOT / 'lab_8_llm' / 'settings.json', 'r', encoding='utf-8') as file:
        result = json.load(file)

    assert result is not None, "Demo does not work correctly"

    return result


if __name__ == "__main__":
    main()

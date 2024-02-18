"""
Check the relevance of stubs.
"""
# pylint: disable=too-many-locals
import sys
from pathlib import Path

from config.constants import PROJECT_CONFIG_PATH
from config.generate_stubs.generator import cleanup_code
from config.generate_stubs.run_generator import format_stub_file, sort_stub_imports
from config.project_config import ProjectConfig


def get_code(code_path: Path) -> str:
    """
    Get clear code from file.

    Args:
        code_path (Path): Path to file with code

    Returns:
        str: Clear code
    """
    with code_path.open(encoding='utf-8') as file:
        code_text = file.read()
    return code_text


def clear_examples(lab_path: Path) -> None:
    """
    Clean temp files.

    Args:
        lab_path (Path): Path to temp files
    """
    example_main_stub_path = lab_path / 'example_main_stub.py'
    example_start_stub_path = lab_path / 'example_start_stub.py'
    example_start_stub_path.unlink()
    example_main_stub_path.unlink()


def main() -> None:
    """
    Check the relevance of stubs.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_paths = project_config.get_labs_paths()
    code_is_equal = True
    for lab_path in labs_paths:
        print(f'Processing {lab_path}...')
        main_stub_path = lab_path / 'main_stub.py'
        start_stub_path = lab_path / 'start_stub.py'

        if not main_stub_path.exists() or not start_stub_path.exists():
            print(f'Ignoring {main_stub_path} or {start_stub_path}: do not exist')
            continue

        main_stub_code = get_code(main_stub_path)
        start_stub_code = get_code(start_stub_path)

        clean_main = cleanup_code(lab_path / 'main.py')
        example_main_stub_path = lab_path / 'example_main_stub.py'
        with example_main_stub_path.open(mode='w', encoding='utf-8') as file:
            file.write(clean_main)
        format_stub_file(example_main_stub_path)
        sort_stub_imports(example_main_stub_path)
        formatted_main = get_code(example_main_stub_path)

        clean_start = cleanup_code(lab_path / 'start.py')
        example_start_stub_path = lab_path / 'example_start_stub.py'
        with example_start_stub_path.open(mode='w', encoding='utf-8') as file:
            file.write(clean_start)
        format_stub_file(example_start_stub_path)
        sort_stub_imports(example_start_stub_path)
        formatted_start = get_code(example_start_stub_path)

        if formatted_main != main_stub_code:
            code_is_equal = False
            print(f'You have different main and main_stub in {lab_path}')

        if formatted_start != start_stub_code:
            code_is_equal = False
            print(f'You have different start and start_stub in {lab_path}')

        if lab_path.name == 'lab_8_llm':
            lab_7_main = get_code(lab_path / 'main.py')
            lab_8_main = get_code(lab_path.parent / 'lab_7_llm' / 'main.py')

            if lab_7_main != lab_8_main:
                code_is_equal = False
                print('You have different main for Lab 7 and Lab 8!')

        clear_examples(lab_path)
    if code_is_equal:
        print('All stubs are relevant')
    sys.exit(not code_is_equal)


if __name__ == '__main__':
    main()

"""
Check docstrings for conformance to the Google-style-docstrings.
"""

import subprocess
import sys
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig


def get_files() -> list:
    """
    Get paths to files in config and core_utils packages.

    Returns:
        list: File paths
    """
    directories = ['config', 'core_utils']
    file_paths = [file for directory in directories
                  for file in Path(PROJECT_ROOT / directory).glob('**/*.py')
                  if file.name != '__init__.py']
    return file_paths


def check_with_pydoctest(path_to_file: Path, path_to_config: Path) \
        -> subprocess.CompletedProcess:
    """
    Check docstrings in file with pydoctest module.

    Args:
        path_to_file (Path): Path to file
        path_to_config (Path): Path to pydoctest config

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    pydoctest_args = [
        '--config',
        str(path_to_config),
        '--file',
        str(path_to_file)
    ]
    return _run_console_tool('pydoctest', pydoctest_args, debug=True)


def check_with_pydocstyle(path_to_file: Path) -> subprocess.CompletedProcess:
    """
    Check docstrings in file with pydocstyle module.

    Args:
        path_to_file (Path): Path to file

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    pydocstyle_args = [
        '-m',
        'pydocstyle',
        str(path_to_file)
    ]
    return _run_console_tool(str(choose_python_exe()), pydocstyle_args, debug=True)


def check_file(path_to_file: Path) -> str:
    """
    Check docstrings in file for conformance to the Google-style-docstrings.

    Args:
        path_to_file (Path): Path to file

    Returns:
        str: Errors in file
    """
    errors = ''
    all_errors = ''
    pydoctest_config = PROJECT_ROOT / 'config' / 'stage_1_style_tests' / 'pydoctest.json'
    pydoctest_checks = check_with_pydoctest(path_to_file, pydoctest_config)
    if pydoctest_checks.returncode == 0:
        print(f'All docstrings in {path_to_file} conform to Google-style'
              f'according to Pydoctest.\n')
    else:
        errors += f'Pydoctest errors:\n{pydoctest_checks.stdout}'

    pydocstyle_checks = check_with_pydocstyle(path_to_file)
    if pydocstyle_checks.returncode == 0:
        print(f'All docstrings in {path_to_file} conform to Google-style'
              f'according to Pydocstyle.\n')
    else:
        errors += f'Pydocstyle errors:\n{pydocstyle_checks.stdout}'

    if errors:
        all_errors += (f'\nDocstrings in {path_to_file} do not conform to Google-style.\n'
                       f'ERRORS:\n{errors}\n')
    return all_errors


def main() -> None:
    """
    Check docstrings for lab, config and core_utils packages.
    """
    all_errors = []
    check_is_good = True
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()
    files_list = get_files()

    # check docstrings in config and core_utils
    for file in files_list:
        print(f'\nChecking {file}')
        all_errors.append(check_file(file))

    # check docstrings in labs
    for lab_path in labs_list:
        main_path = lab_path / 'main.py'

        if not main_path.exists():
            print(f'\nIgnoring {main_path}: it does not exist.')
            continue
        print(f'\nChecking {main_path}')

        all_errors.append(check_file(main_path))

    if not all(el == '' for el in all_errors):
        check_is_good = False
        print('\n'.join(all_errors))
        print('\nThe docstring check was not successful!')

    sys.exit(not check_is_good)


if __name__ == '__main__':
    main()

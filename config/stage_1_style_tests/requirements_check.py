"""
Checks dependencies
"""
import re
import sys
from pathlib import Path

from config.constants import PROJECT_ROOT


def get_paths() -> list[Path]:
    """
    Returns list of paths to non-python files
    """
    return list(PROJECT_ROOT.rglob('requirements*.txt'))


def get_requirements(path: Path) -> list:
    """
    Returns a list of dependencies
    """
    with path.open(encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def compile_pattern() -> re.Pattern:
    """
    Returns the compiled pattern
    """
    return re.compile(r'((\w+(-\w+|\[\w+\])*==\d+(\.\d+)+)'
                      r'|((-r|--extra-index-url)\s.*))', re.MULTILINE)


def check_dependencies(lines: list, compiled_pattern: re.Pattern, path: Path) -> bool:
    """
    Checks that dependencies confirm to the template
    """
    expected = [
        i
        for i in sorted(map(str.lower, lines))
        if i.split()[0] not in ('--extra-index-url',)
    ]
    actual = [
        i
        for i in map(str.lower, lines)
        if i.split()[0] not in ('--extra-index-url',)
    ]
    if expected != actual:
        print(f'Dependencies in {path.relative_to(PROJECT_ROOT)} do not follow sorting rule.')
        print('Expected:')
        print('\n'.join(expected))
        return False
    for line in lines:
        if not re.search(compiled_pattern, line):
            print(f'Specific dependency in {path.relative_to(PROJECT_ROOT)} '
                  'do not conform to the template.')
            print(line)
            return False
    return True


def main() -> None:
    """
    Calls functions
    """
    paths = get_paths()
    compiled_pattern = compile_pattern()
    for path in paths:
        lines = get_requirements(path)
        if not check_dependencies(lines, compiled_pattern, path):
            print(f'{path.relative_to(PROJECT_ROOT)} : FAIL')
            sys.exit(1)
        else:
            print(f'{path.relative_to(PROJECT_ROOT)} : OK')


if __name__ == '__main__':
    main()

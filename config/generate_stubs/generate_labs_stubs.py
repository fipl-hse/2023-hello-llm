"""
Generator of all labs.
"""

from pathlib import Path

from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.generate_stubs.generator import cleanup_code
from config.generate_stubs.run_generator import format_stub_file, sort_stub_imports
from config.project_config import ProjectConfig


def _generate_stubs_single_module(module_path: Path) -> None:
    """
    Process single module.

    Args:
        module_path (Path): Path to module
    """
    stub_path = module_path.parent / f'{module_path.stem}_stub{module_path.suffix}'

    source_code = cleanup_code(module_path)
    with stub_path.open(mode='w', encoding='utf-8') as f:
        f.write(source_code)
    format_stub_file(stub_path)
    sort_stub_imports(stub_path)


def generate_all_stubs(project_config: ProjectConfig) -> None:
    """
    Generate stubs for all labs.

    Args:
        project_config (ProjectConfig): Project config
    """
    labs = project_config.get_labs_names()
    for lab_name in labs:
        print(f'Generating stubs for {lab_name}')
        module_paths = (
                PROJECT_ROOT / lab_name / 'main.py',
                PROJECT_ROOT / lab_name / 'start.py',
                PROJECT_ROOT / lab_name / 'service.py',
        )
        for module_path in module_paths:
            if not module_path.exists():
                continue
            print(f'\t\t{module_path}')
            _generate_stubs_single_module(module_path)


def main() -> None:
    """
    Entrypoint for stub generation.
    """
    proj_conf = ProjectConfig(PROJECT_CONFIG_PATH)
    generate_all_stubs(proj_conf)


if __name__ == '__main__':
    main()

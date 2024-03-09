"""
Generator of stubs for existing lab implementation.
"""
# pylint: disable=too-many-statements
import ast
from _ast import alias, stmt
from pathlib import Path
from typing import Optional

import ast_comments
from tap import Tap


class NoDocStringForAMethodError(Exception):
    """
    Error for a method that lacks docstring.
    """


def remove_implementation_from_function(original_declaration: ast.stmt,
                                        parent: Optional[ast.ClassDef] = None) -> None:
    """
    Remove reference implementation.

    Args:
        original_declaration (ast.stmt): Original declaration
        parent (Optional[ast.ClassDef]): Parent
    """
    if not isinstance(original_declaration, ast.FunctionDef):
        return
    # print(ast.dump(ast.parse("raise NotImplementedError()"), annotate_fields=False, indent=4))
    expr = original_declaration.body[0]
    if not isinstance(expr, ast.Expr) and \
            (
                    not hasattr(expr, 'value') or
                    not isinstance(getattr(expr, 'value'), ast.Constant)
            ):
        raise NoDocStringForAMethodError(f'You have to provide docstring for a method '
                                         f'{parent.name + "." if parent is not None else ""}'
                                         f'{original_declaration.name}')

    opening_files = []
    for decl in original_declaration.body:
        if isinstance(decl, ast.With):
            if 'assets' in ast.unparse(decl.items[0].context_expr.args[0]):  # type: ignore
                opening_files.append(decl)

        if isinstance(decl, ast.Assert):
            add_none = ast.parse("result = None")
            opening_files.extend([add_none, decl])  # type: ignore
    original_declaration.body[1:] = opening_files


# pylint: disable=too-many-branches
def cleanup_code(source_code_path: Path) -> str:
    """
    Remove implementation based on AST parsing of code.

    Args:
        source_code_path (Path): Path to source code

    Returns:
        str: Implementation without AST parsing of code
    """
    with source_code_path.open(encoding='utf-8') as file:
        data = ast.parse(file.read(), source_code_path.name, type_comments=True)

    with source_code_path.open(encoding='utf-8') as file:
        data_2 = ast_comments.parse(file.read(), source_code_path.name)

    accepted_modules: dict[str, list[str]] = {
        'typing': ['*'],
        'torch.utils.data': ['Dataset'],
        'lab_3_generate_by_ngrams.main': ['*'],
        'core_utils.llm.time_decorator': ['report_time'],
    }

    if source_code_path.name not in ('start.py', 'service.py'):
        accepted_modules['pathlib'] = ['Path']
        accepted_modules['torch'] = ['']
        accepted_modules['pandas'] = ['DataFrame']
        accepted_modules['core_utils.llm.llm_pipeline'] = ['AbstractLLMPipeline']
        accepted_modules['core_utils.llm.metrics'] = ['Metrics']
        accepted_modules['core_utils.llm.raw_data_importer'] = ['AbstractRawDataImporter']
        accepted_modules['core_utils.llm.raw_data_preprocessor'] = ['AbstractRawDataPreprocessor']
        accepted_modules['core_utils.llm.task_evaluator'] = ['AbstractTaskEvaluator']

    if 'reference_lab' in source_code_path.parent.name:
        accepted_modules['config.lab_settings'] = ['InferenceParams']

    if source_code_path.name == 'start.py' and 'llm' in source_code_path.parent.name:
        accepted_modules.pop('typing')

    if source_code_path.name == 'main.py' and 'reference_lab' in source_code_path.parent.name:
        accepted_modules.pop('pandas')

    if source_code_path.name == 'service.py':
        accepted_modules['fastapi'] = ['FastAPI']
        accepted_modules['lab_7_llm.main'] = ['LLMPipeline']

    new_decl: list[stmt] = []

    for decl_2 in data_2.body:
        if isinstance(decl_2, ast_comments.Comment):
            data.body.insert(data_2.body.index(decl_2), decl_2)

    for decl in data.body:
        if isinstance(decl, ast.AsyncFunctionDef) or \
                isinstance(decl, ast.ClassDef) and \
                decl.name == 'Query':
            decl = []  # type: ignore

        if source_code_path.name == 'service.py' and \
                isinstance(decl, ast.Assign):
            decl = ast.parse("app, pipeline = None, None")  # type: ignore

        if isinstance(decl, (ast.Import, ast.ImportFrom)):
            if (module_name := getattr(decl, 'module', None)) is None:
                module_name = decl.names[0].name

            if module_name not in accepted_modules:
                continue

            if isinstance(decl, ast.ImportFrom):
                accepted_names = accepted_modules.get(module_name, [])
                names_to_import = [name for name in decl.names if name.name
                                   in accepted_names or '*' in accepted_names]

                if not names_to_import:
                    continue

                new_decl.append(ast.ImportFrom(module=module_name,
                                               names=[alias(name=name.name)
                                                      for name in names_to_import]))
                continue

        if isinstance(decl, ast.ClassDef) and \
                isinstance(ast.get_docstring(decl), str):
            if 'Note: remove' in ast.get_docstring(decl):  # type: ignore
                decl = []  # type: ignore
            else:
                for class_decl in decl.body:
                    if not isinstance(class_decl, ast.FunctionDef):
                        continue

                    docstring = ast.get_docstring(class_decl)
                    if docstring is None:
                        raise ValueError(
                            f'{source_code_path.parent.name}.{source_code_path.stem}.'
                            f'{decl.name}.{class_decl.name} does not have a docstring!'
                        )

                    if 'Note: remove' in ast.get_docstring(class_decl):  # type: ignore
                        decl.body[decl.body.index(class_decl)] = []  # type: ignore

        if isinstance(decl, ast.ClassDef) and decl.bases:
            name = decl.bases[0]
            if decl.bases and isinstance(name, ast.Name) and \
                    hasattr(name, 'id') and \
                    getattr(name, 'id') == 'Exception':
                decl = []  # type: ignore

        if isinstance(decl, ast.ClassDef):
            for class_decl in decl.body:
                remove_implementation_from_function(class_decl, parent=decl)

        remove_implementation_from_function(decl)

        new_decl.append(decl)

    data.body = list(new_decl)
    return ast_comments.unparse(data)  # type: ignore


class ArgumentParser(Tap):
    """
    Types for CLI interface of a module.
    """
    source_code_path: str
    target_code_path: str


def main() -> None:
    """
    Entrypoint for stub generation.
    """
    args = ArgumentParser().parse_args()

    res_stub_path = Path(args.target_code_path)
    res_stub_path.parent.mkdir(parents=True, exist_ok=True)

    source_code = cleanup_code(Path(args.source_code_path))

    with res_stub_path.open(mode='w', encoding='utf-8') as file:
        print(f'Writing to {res_stub_path}')
        file.write(source_code)


if __name__ == '__main__':
    main()

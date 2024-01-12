set -x

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "core_utils"
    "lab_7_llm"
  )
else
  DIRS_TO_CHECK=(
    "reference_lab_nmt"
    "reference_lab_nli"
    "reference_lab_generation"
    "reference_lab_classification"
    "reference_lab_summarization"
    "lab_7_llm"
    "config"
    "seminars"
    "core_utils"
  )
fi

python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

if [[ "$1" != "smoke" ]]; then
  python -m pydocstyle reference_lab_generation/main.py
  pydoctest --config config/stage_1_style_tests/pydoctest.json --file reference_lab_generation/main.py
  pydoctest --config config/stage_1_style_tests/pydoctest.json --file reference_lab_generation/service.py

  python -m pytest -m "mark10"
fi


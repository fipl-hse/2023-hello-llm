"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchinfo import summary

try:
    import torch
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = dict
    torch = namedtuple('torch', 'no_grad')(lambda: lambda fn: fn)  # type: ignore

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        dataset_df = load_dataset(self._hf_name, split='validation').to_pandas()
        self._raw_data = dataset_df

        if type(self._raw_data) is not DataFrame:
            raise TypeError


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        dataset_analysis = {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': len(self._raw_data[self._raw_data.duplicated()]),
            'dataset_empty_rows': len(self._raw_data[self._raw_data.isna().any(axis=1)])}

        self._raw_data = self._raw_data.dropna()

        dataset_analysis['dataset_sample_min_len'] = len(min(self._raw_data['report'], key=len))
        dataset_analysis['dataset_sample_max_len'] = len(max(self._raw_data['report'], key=len))

        return dataset_analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
            self,
            model_name: str,
            dataset: TaskDataset,
            max_length: int,
            batch_size: int,
            device: str
    ) -> None:
        """
        Initialize an instance of LLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """

        tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization")
        text = "The structure of the armed forces is based on the Total Force concept, which recognizes that all elements of the structure—active duty military personnel, reservists, defense contractors, host nation military and civilian personnel, and DOD federal civilian employees—contribute to national defense. In recent years, federal civilian personnel have deployed along with military personnel to participate in Operations Joint Endeavor, conducted in the countries of Bosnia-Herzegovina, Croatia, and Hungary; Joint Guardian, in Kosovo; and Desert Storm, in Southwest Asia. Further, since the beginning of the Global War on Terrorism, the role of DOD’s federal civilian personnel has expanded to include participation in combat support functions in Operations Enduring Freedom and Iraqi Freedom. DOD relies on the federal civilian personnel it deploys to support a range of essential missions, including intelligence collection, criminal investigations, and weapon systems acquisition and maintenance. To ensure that its federal civilian employees will deploy to combat zones and perform critical combat support functions in theater, DOD established the emergency-essential program in 1985. Under this program, DOD designates as “emergency-essential” those civilian employees whose positions are required to ensure the success of combat operations or the availability of combat-essential systems. DOD can deploy federal civilian employees either on a voluntary or involuntary basis to accomplish the DOD mission. DOD has established force health protection and surveillance policies aimed at assessing and reducing or preventing health risks for its deployed federal civilian personnel; however, the department lacks procedures to ensure the components’ full implementation of its policies. In reviewing DOD federal civilian deployment records and other electronic documentation at selected component locations, we found that these components lacked documentation to show that they had fully complied with DOD’s force health protection and surveillance policy requirements for some federal civilian personnel who deployed to Afghanistan and Iraq. As a larger issue, DOD’s policies did not require the centralized collection of data on the identity of its deployed civilians, their movements in theater, or their health status, further hindering its efforts to assess the overall effectiveness of its force health protection and surveillance capabilities. In August 2006, DOD issued a revised policy (to be effective in December 2006) that outlines procedures to address its lack of centralized deployment and health-related data. However, the procedures are not comprehensive enough to ensure that DOD will be sufficiently informed of the extent to which its components fully comply with its requirements to monitor the health of deployed federal civilians. The DOD components included in our review lacked documentation to show that they always implemented force health protection and surveillance requirements for deployed federal civilians. These requirements include completing (1) pre-deployment health assessments to ensure that only medically fit personnel deploy outside of the United States as part of a contingency or combat operation; (2) pre-deployment immunizations to address possible health threats in deployment locations; (3) pre-deployment medical screenings for tuberculosis and human immunodeficiency virus (HIV); and (4) post-deployment health assessments to document current health status, experiences, environmental exposures, and health concerns related to their work while deployed. DOD’s force health protection and surveillance policies require the components to assess the medical condition of federal civilians to ensure that only medically fit personnel deploy outside of the United States as part of a contingency or combat operation. The policies stipulate that all deploying civilian personnel are to complete pre-deployment health assessment forms within 30 days of their deployments, and health care providers are to review the assessments to confirm the civilians’ health readiness status and identify any needs for additional clinical evaluations prior to their deployments. While the components that we included in our review had procedures in place that would enable them to implement DOD’s pre-deployment health assessment policies, it was not clear to what extent they had done so. Our review of deployment records and other documentation at the selected component locations found that these components lacked documentation to show that some federal civilian personnel who deployed to Afghanistan and Iraq had received the required pre-deployment health assessments. For those deployed federal civilians in our review, we found that, overall, a small number of deployment records (52 out of 3,771) were missing documentation to show that they had received their pre-deployment health assessments, as reflected in table 1. As shown in table 1, the federal civilian deployment records we included in our review showed wide variation by location regarding documentation of pre-deployment health assessments, ranging from less than 1 percent to more than 90 percent. On an aggregate component-level basis, at the Navy location in our review, we found that documentation was missing for 19 of the 52 records in our review. At the Air Force locations, documentation was missing for 29 of the 37 records in our review. In contrast, all three Army locations had hard copy or electronic records which indicated that almost all of their federal deployed civilians had received pre-deployment health assessments. In addition to completing pre-deployment health assessment forms, DOD’s force health protection and surveillance policies stipulate that all DOD deploying federal civilians receive theater-specific immunizations to address possible health threats in deployment locations. Immunizations required for all civilian personnel who deploy to Afghanistan and Iraq include: hepatitis A (two-shot series); tetanus-diphtheria (within 10 years of deployment); smallpox (within 5 years of deployment); typhoid; and influenza (within the last 12 months of deployment). As reflected in table 2, based on the deployment records maintained by the components at locations included in our review, the overall number of federal civilian deployment records lacking documentation of only one of the required immunizations for deployment to Afghanistan and Iraq was 285 out of 3,771. However, 3,313 of the records we reviewed were missing documentation of two or more immunizations. At the Army’s Fort Bliss, our review of its electronic deployment data determined that none of its deployed federal civilians had documentation to show that they had received immunizations. Officials at this location stated that they believed some immunizations had been given; however, they could not provide documentation as evidence of this. DOD policies require deploying federal civilians to receive certain screenings, such as for tuberculosis and HIV. Table 3 indicates that 55 of the 3,771 federal civilian deployment records included in our review were lacking documentation of the required tuberculosis screening; and approximately 35 were lacking documentation of HIV screenings prior to deployment. DOD’s force health protection and surveillance policies also require returning DOD federal civilian personnel to undergo post-deployment health assessments to document current health status, experiences, environmental exposures, and health concerns related to their work while deployed. The post-deployment process begins within 5 days of civilians’ redeployment from the theater to their home or demobilization processing stations. DOD’s policies require civilian personnel to complete a post- deployment assessment that includes questions on health and exposure concerns. A health care provider is to review each assessment and recommend additional clinical evaluation or treatment as needed. As reflected in table 4, our review of deployment records at the selected component locations found that these components lacked documentation to show that most deployed federal civilians (3,525 out of 3,771) who deployed to Afghanistan and Iraq had received the required post- deployment health assessments upon their return to the United States. Federal civilian deployment records lacking evidence of post-deployment health assessments ranged from 3 at the U.S. Army Corps of Engineers Transatlantic Programs Center and Wright-Patterson Air Force Base, respectively, to 2,977 at Fort Bliss. Beyond the aforementioned weaknesses found in the selected components’ implementation of force health protection and surveillance requirements for deploying federal civilians, as a larger issue, DOD lacks comprehensive, centralized data that would enable it to readily identify its deployed civilians, track their movements in theater, or monitor their health status, further hindering efforts to assess the overall effectiveness of its force health protection and surveillance capabilities. The Defense Manpower Data Center (DMDC) is responsible for maintaining the department’s centralized system that currently collects location-specific deployment information for military servicemembers, such as grid coordinates, latitude/longitude coordinates, or geographic location codes. However, DOD has not"
        tokens = tokenizer(text, return_tensors="pt")
        print(tokens.keys())
        tensor_data = torch.ones(1, 512, dtype=torch.long)
        input_data = {"input_ids": tensor_data, "token_type_ids": tensor_data, "attention_mask": tensor_data}
        model_statistics = summary(model, input_data=input_data, verbose=False)
        print(model_statistics)
        total_params = model_statistics.total_params
        trainable_params = model_statistics.trainable_params
        last_layer = model_statistics.summary_list[-1].output_size
        print(total_params, trainable_params, last_layer)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """

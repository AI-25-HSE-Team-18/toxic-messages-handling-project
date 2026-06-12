"""
Loads and runs in parallel mode every predictor listed in config 
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from services.model import LinearSVMModel, BertClassifierModel, Model
from services.utils import load_config
from core.config import MODEL_CONFIG
from services.model import MODEL_INFERENCE_DURATION, MODEL_INFERENCE_TOTAL

logger = logging.getLogger(__name__)

# map config "model_type" → Model subclass
PREDICTOR_TYPE_REGISTRY = {
    "linear_svm": LinearSVMModel,
    "bert": BertClassifierModel,
}


class ModelRegistry:
    """
    Loads all predictors defined in config['predictors'] and exposes
    `run_all(text)` for parallel multi-model inference.
    """

    def __init__(self, config_path: str = MODEL_CONFIG):
        self._models: List[Model] = []
        
        # One worker thread per model keeps things simple and avoids GIL contention
        self._executor = ThreadPoolExecutor(
            max_workers=None,  # defaults to (cpu_count or 1) * 5
            thread_name_prefix="model_worker",
        )
        self._load_all(config_path)

    def _load_all(self, config_path: str) -> None:
        config = load_config(config_path)
        predictors = config.get("predictors", [])

        for i, predictor_conf in enumerate(predictors):
            model_type = predictor_conf.get("model_type", "linear_svm")
            cls = PREDICTOR_TYPE_REGISTRY.get(model_type)
            if cls is None:
                raise ValueError(
                    f"Unknown model_type '{model_type}' at predictor index {i}. "
                    f"Available: {list(PREDICTOR_TYPE_REGISTRY)}"
                )
            logger.info("Loading model[%d] type=%s …", i, model_type)
            model = cls(config_path=config_path, worker_id=i)
            self._models.append(model)
            logger.info("Model[%d] '%s' ready.", i, model.model_id)

        # Pre-initialize Prometheus label combinations for every loaded model so
        # they appear in label_values() immediately — before any request arrives.
        for model in self._models:
            MODEL_INFERENCE_DURATION.labels(model_id=model.model_id)
            for lbl in ("toxic", "non_toxic"):
                MODEL_INFERENCE_TOTAL.labels(model_id=model.model_id, prediction_label=lbl)

    @property
    def models(self) -> List[Model]:
        return self._models

    async def run_all(self, text: str) -> List[Tuple[str, int, str, float]]:
        """
        Run all models in parallel.

        Returns a list of (model_id, prediction_int, prediction_label, processing_time_ms)
        for each mdel
        """
        loop = asyncio.get_event_loop()
        tasks = [
            # loop.run_in_executor(self._executor, model.predict_full, text)
            loop.run_in_executor(self._executor, model.predict_log_prometheus, text)
            for model in self._models
        ]
        results: List[Tuple[str, int, str, float]] = await asyncio.gather(*tasks)
        return results

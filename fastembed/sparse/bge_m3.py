from typing import Any, Iterable, Sequence, Type
import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.model_description import SparseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import Device
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import SparseEmbedding, SparseTextEmbeddingBase,
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

supported_bge_m3_models: list[SparseModelDescription] = [
    SparseModelDescription(
        model="BAAI/bge-m3",
        vocab_size=250002,
        description=(
            "Sparse embedding model with lexical weights, Multilingual (100+ languages), "
            "8192 input tokens truncation, "
            "generates sparse token-level weights via learned linear projection and ReLU."
        ),
        license="mit",
        size_in_GB=2.27,
        sources=ModelSource(hf="aapot/bge-m3-onnx"),
        model_file="model.onnx",
        additional_files=["onnx/model.onnx_data", "onnx/sentencepiece.bpe.model"],
    ),
]


class BgeM3(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    """BGE-M3 sparse embedding model.
    Multi-Functionality: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval.

    Uses the lexical weight output from BGE-M3 (the second output of the
    multi-output ONNX model). The model produces per-token weights via
    a learned linear projection followed by ReLU activation over the
    encoder hidden states.
    """
    ONNX_OUTPUT_NAMES: list[str] | None = None

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        if output.input_ids is None:
            raise ValueError("input_ids must be provided for BGE-M3 sparse post-processing")

        if output.metadata is None or "sparse_vecs" not in output.metadata:
            raise ValueError(
                "sparse_vecs not found in output metadata. "
                "Ensure the ONNX model provides sparse output."
            )

        sparse_vecs = output.metadata["sparse_vecs"]  
        token_weights = sparse_vecs.squeeze(-1) 

        unused_tokens = set()
        for name in ("cls", "eos", "pad", "unk"):
            token_id = self.special_token_to_id.get(f"<{name}>") or self.special_token_to_id.get(
                f"[{name.upper()}]"
            )
            if token_id is not None:
                unused_tokens.add(token_id)

        for weights, input_ids in zip(token_weights, output.input_ids):
            token_weight_map: dict[int, float] = {}
            for w, idx in zip(weights, input_ids):
                if idx in unused_tokens or w <= 0:
                    continue
                idx_int = int(idx)
                if idx_int not in token_weight_map or w > token_weight_map[idx_int]:
                    token_weight_map[idx_int] = float(w)

            if token_weight_map:
                indices = np.array(list(token_weight_map.keys()), dtype=np.int32)
                values = np.array(list(token_weight_map.values()), dtype=np.float32)
            else:
                indices = np.array([], dtype=np.int32)
                values = np.array([], dtype=np.float32)

            yield SparseEmbedding(values=values, indices=indices)

    def onnx_embed(self, documents: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Override to capture all ONNX outputs including sparse_vecs."""
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])
        input_names = {node.name for node in self.model.get_inputs()}  
        onnx_input: dict[str, np.ndarray] = {
            "input_ids": np.array(input_ids, dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(attention_mask, dtype=np.int64)
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            )
        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)

        model_output = self.model.run(None, onnx_input)  

        return OnnxOutputContext(
            model_output=model_output[0],  # dense_vecs
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
            metadata={"sparse_vecs": model_output[1]},  # sparse token weights
        )

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        return self._token_count(texts, batch_size=batch_size, **kwargs)

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        return supported_bge_m3_models

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)

        self.device_ids = device_ids
        self.cuda = cuda

        self.device_id: int | None = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))

        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        if not self.lazy_load:
            self.load_onnx_model()

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
        )

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of sparse embeddings.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of sparse embeddings, one per document
        """
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker[SparseEmbedding]]:
        return BgeM3EmbeddingWorker


class BgeM3EmbeddingWorker(TextEmbeddingWorker[SparseEmbedding]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> BgeM3:
        return BgeM3(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )

"""
# ====================================================== #
#               Adaptation of module                     #
# ------------------------------------------------------ #
#           original module source below:                #
#        https://github.com/matheusaf/aiNet             #
# ====================================================== #
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import spacy
from numba import from_dtype, jit, prange
from numba.core import types
from numba.typed import Dict
from pandas import DataFrame
from spacy import parts_of_speech
from spacy.lang.en import English

from .representation import Representation

# =========================================================
# NUMBA TYPES
# =========================================================

INT_TYPE = types.int64

UNICHAR_TYPE = from_dtype(np.dtype("<U5"))

# =========================================================
# HELPER NUMBA
# =========================================================


@jit(
    nopython=True,
    cache=True,
)
def generate_representation_helper(
    tagged_sentence: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """
    Retorna vetor de frequência normalizado das POS tags.
    """

    tag_counter: Dict = Dict.empty(
        key_type=UNICHAR_TYPE,
        value_type=INT_TYPE,
    )

    feature_size: int = features.shape[0]

    representation_row = np.zeros(
        feature_size,
        dtype=np.float32,
    )

    tags_size: int = tagged_sentence.shape[0]

    # =====================================================
    # CONTAGEM TAGS
    # =====================================================

    for i in range(tags_size):

        tag = tagged_sentence[i]

        if tag not in tag_counter:
            tag_counter[tag] = 0

        tag_counter[tag] += 1

    # =====================================================
    # GERA VETOR
    # =====================================================

    for i in range(feature_size):

        feature = features[i]

        col_counter = tag_counter.get(feature)

        if col_counter is not None:
            representation_row[i] = col_counter

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    if tags_size > 0:
        return representation_row / tags_size

    return representation_row


# =========================================================
# STAGGER
# =========================================================


class STagger(Representation):
    """
    POS-Tagging usando spaCy puro
    """

    _spacy_model: English | Any

    def __init__(
        self,
        spacy_model_name: str = "en_core_web_sm",
    ) -> None:

        super().__init__(stop_word_removal_enabled=False)

        # =================================================
        # VERIFICA MODELO
        # =================================================

        try:

            self._spacy_model = spacy.load(
                spacy_model_name,
                disable=[
                    "parser",
                    "ner",
                    "lemmatizer",
                    "textcat",
                ],
            )

        except Exception as e:

            raise RuntimeError(
                f"\nErro carregando modelo spaCy "
                f"'{spacy_model_name}'\n\n"
                f"Erro original:\n{str(e)}"
            )

        # =================================================
        # FORÇA CPU
        # =================================================

        spacy.require_cpu()

        # =================================================
        # LOAD MODELO
        # =================================================

        self._spacy_model = spacy.load(
            spacy_model_name,
            disable=[
                "parser",
                "ner",
                "lemmatizer",
                "textcat",
            ],
        )

        # =================================================
        # FEATURES POS
        # =================================================

        self.features = np.array(
            sorted(parts_of_speech.IDS.keys())[1:],
            dtype="<U5",
        )

        self._representation = np.array([])

    # =====================================================
    # TAGGING PIPE
    # =====================================================

    def run_tagging_pipe(
        self,
        sentences: list[str] | str,
    ) -> Generator:
        """
        Retorna POS tags por sentença.
        """

        assert isinstance(sentences, list) or isinstance(sentences, str), (
            "The sentences must be " "a list of strings or a string."
        )

        tagging_sentences = sentences

        if isinstance(sentences, str):

            tagging_sentences = [sentences]

        # =================================================
        # PREPROCESSING FRAMEWORK
        # =================================================

        processed_sentences = list(self.pre_process(tagging_sentences))

        # =================================================
        # PIPE
        # =================================================

        docs = self._spacy_model.pipe(
            processed_sentences,
            batch_size=256,
            n_process=1,
        )

        # =================================================
        # TAGS
        # =================================================

        for doc in docs:

            yield np.array(
                [token.pos_ for token in doc],
                dtype="<U5",
            )

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def generate_representation(
        self,
        sentences: list[str],
        as_dataframe: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        """
        Retorna matriz POS-tag frequency.
        """

        # =================================================
        # CPU
        # =================================================

        spacy.require_cpu()

        # =================================================
        # TAGGING
        # =================================================

        tagged_sentences: list[np.ndarray] = list(self.run_tagging_pipe(sentences))

        sentence_count: int = len(tagged_sentences)

        # =================================================
        # MATRIZ FINAL
        # =================================================

        self._representation = np.zeros(
            (
                sentence_count,
                self._features.shape[0],
            ),
            dtype=np.float32,
        )

        # =================================================
        # REPRESENTAÇÃO
        # =================================================

        for sentence_index in prange(sentence_count):

            tagged_sentence = tagged_sentences[sentence_index]

            self._representation[sentence_index] = generate_representation_helper(
                tagged_sentence,
                self._features,
            )

        # =================================================
        # DATAFRAME
        # =================================================

        if as_dataframe:

            return DataFrame(
                data=self._representation,
                columns=self._features,
            )

        return (
            self.features,
            self.representation,
        )


# =========================================================
# REGISTER
# =========================================================


# =========================================================
# TESTE
# =========================================================

if __name__ == "__main__":

    s = STagger()

    data = (
        [
            "Use this metaclass to create an ABC.",
            "An ABC can be subclassed directly, and then acts as a mix-in class.",
            """
            You can also register unrelated concrete classes
            (even built-in classes) and unrelated ABCs as
            virtual subclasses.
            """,
        ]
        * 10
    )

    df = s.generate_representation(
        data,
        as_dataframe=True,
    )

    print(df)

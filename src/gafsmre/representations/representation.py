"""
# ====================================================== #
#               Adaptation of module                     #
# ------------------------------------------------------ #
#           original module source below:                #
#        https://github.com/matheusaf/aiNet             #
# ====================================================== #
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Optional

import numpy as np
from pandas import DataFrame
from spacy.language import Language

import gafsmre.utils.text.processing as text_processing


class Representation(metaclass=ABCMeta):
    """
    Abstract class for all representations.
    """

    __slots__ = [
        "_representation",
        "_features",
        "_spacy_model",
        "_is_stop_word_removal_enabled",
        "_stop_words",
    ]

    _features: np.ndarray
    _spacy_model: Optional[Language]
    _representation: np.ndarray
    _is_stop_word_removal_enabled: bool
    _stop_words: Optional[set[str]]

    # =====================================================
    # INIT
    # =====================================================

    def __init__(
        self,
        stop_word_removal_enabled: bool = False,
    ) -> None:

        self._is_stop_word_removal_enabled = (
            stop_word_removal_enabled
        )

        self._stop_words = None

        self._spacy_model = None

        self._representation = np.array([])

        self._features = np.array([])

    # =====================================================
    # STOP WORDS
    # =====================================================

    @property
    def stop_words(self) -> set[str] | None:
        """
        :return: stop words
        """

        if self._stop_words is not None:

            return {*self._stop_words}

        return self._stop_words

    @stop_words.setter
    def stop_words(
        self,
        value: Optional[set[str]],
    ) -> None:
        """
        stop words setter
        """

        self._stop_words = value

    # =====================================================
    # FEATURES
    # =====================================================

    @property
    def features(self) -> np.ndarray:
        """
        :return: features
        """

        return self._features

    @features.setter
    def features(
        self,
        new_features: np.ndarray | list[str],
    ) -> None:

        self._representation = np.zeros(
            (len(new_features), 1)
        )

        if isinstance(new_features, list):

            self._features = np.array(
                new_features
            )

            return

        self._features = new_features

    # =====================================================
    # REPRESENTATION
    # =====================================================

    @property
    def representation(self) -> np.ndarray:
        """
        :return: representation matrix
        """

        return self._representation

    @representation.setter
    def representation(
        self,
        value: np.ndarray,
    ) -> None:
        """
        representation setter
        """

        assert isinstance(value, np.ndarray)

        self._representation = value

    # =====================================================
    # FLAGS
    # =====================================================

    @property
    def remove_stop_words_flag(self) -> bool:
        """
        stop word flag
        """

        return self._is_stop_word_removal_enabled

    # =====================================================
    # ABSTRACT
    # =====================================================

    @abstractmethod
    def generate_representation(
        self,
        sentences: list[str],
        as_dataframe: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        """
        Generate representation
        """

        raise NotImplementedError()

    # =====================================================
    # PRE PROCESS
    # =====================================================

    def pre_process(
        self,
        sentences: list[str],
    ) -> Iterator[str]:
        """
        Pre process list
        """

        assert isinstance(sentences, list) and all(
            isinstance(sentence, str)
            for sentence in sentences
        )

        for sentence in sentences:

            yield self._pre_process_sentence(
                sentence
            )

    # =====================================================
    # PRE PROCESS SINGLE
    # =====================================================

    def _pre_process_sentence(
        self,
        sentence: str,
    ) -> str:
        """
        Pre process single sentence
        """

        assert isinstance(sentence, str)

        final_sentence = sentence.casefold()

        final_sentence = (
            text_processing.remove_accents(
                final_sentence
            )
        )

        final_sentence = (
            text_processing.remove_special_chars(
                final_sentence
            )
        )

        final_sentence = (
            text_processing.remove_additional_spaces(
                final_sentence
            )
        )

        final_sentence = (
            text_processing.remove_start_end_space(
                final_sentence
            )
        )

        return (
            text_processing.remove_additional_quotations(
                final_sentence
            )
        )

    # =====================================================
    # STOP WORD REMOVAL
    # =====================================================

    def remove_stop_words(
        self,
        sentences: list[str],
    ) -> list[str] | np.ndarray:
        """
        Remove stop words
        """

        assert isinstance(sentences, list) and all(
            isinstance(sentence, str)
            for sentence in sentences
        )

        if (
            not self._is_stop_word_removal_enabled
            or self._spacy_model is None
            or self._stop_words is None
        ):

            return sentences

        docs = self._spacy_model.pipe(
            sentences,
            disable=[
                "ner",
                "textcat",
                "parser",
            ],
            n_process=1,
            batch_size=256,
        )

        return [
            " ".join(
                [
                    valid_token
                    for token in doc
                    if (
                        valid_token := (
                            token.lemma_.casefold()
                        )
                    ) not in self._stop_words
                ]
            )
            for doc in docs
        ]
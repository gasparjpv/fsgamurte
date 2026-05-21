"""
# ====================================================== #
#               Adaptation of module                     #
# ------------------------------------------------------ #
#           original module source below:                #
#        https://github.com/matheusaf/aiNet             #
# ====================================================== #
"""

from __future__ import annotations

import os
import pathlib
import pickle
from collections.abc import Callable, Generator

import numpy as np
from pandas import DataFrame

from .stagger import STagger
from .representation import Representation

from gafsmre.utils.text.processing import (
    remove_all_spaces,
    remove_punctuations,
)


class MRC2(Representation):
    """
    MRC2 lexical representation
    """

    # =====================================================
    # TYPES
    # =====================================================

    _mrc_dic: dict

    _stagger: object

    _dic_filepath: str | pathlib.Path | None

    _pickle_filepath: str | pathlib.Path

    # =====================================================
    # NUMERICAL RANGES
    # =====================================================

    __numerical_ranges_ = [
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 10),
        (10, 12),
        (12, 15),
        (15, 21),
        (21, 25),
        (25, 28),
        (28, 31),
        (31, 34),
        (34, 37),
        (37, 40),
        (40, 43),
    ]

    # =====================================================
    # FEATURES
    # =====================================================

    __numerical_category_names_ = [
        "NLET",
        "NPHON",
        "NSYL",
        "K-F-FREQ",
        "K-F-NCATS",
        "K-F-NSAMP",
        "T-L-FREQ",
        "BROWN-FREQ",
        "FAM",
        "CONC",
        "IMAG",
        "MEANC",
        "MEANP",
        "AOA",
        "DERIVATIONAL",
        "ABBREVIATION",
        "SUFFIX",
        "PREFIX",
        "HYPHENATED",
        "MULTI-WORD",
        "DIALECT",
        "ALIEN",
        "ARCHAIC",
        "COLLOQUIAL",
        "CAPITAL",
        "ERRONEOUS",
        "NONSENSE",
        "NONCE WORD",
        "OBSOLETE",
        "POETICAL",
        "RARE",
        "RHETORICAL",
        "SPECIALISED",
        "STANDARD",
        "SUBSTANDARD",
        "PRONUNCIATION_DIFFER_STRESS",
        "PRONUNCIATION_DIFFER",
        "CAPITALIZATION",
        "PLURAL",
        "SINGULAR",
        "BOTH_SINGULAR_PLURAL",
        "NO_PLURAL",
        "PLURAL_ACT_SINGULAR",
    ]

    # =====================================================
    # POS MAP
    # =====================================================

    __universal_tag_to_wtype_ = {
        "PRON": ("U",),
        "VERB": ("V", "P"),
        "DET": ("J",),
        "NOUN": ("N",),
        "ADP": ("R",),
        "ADJ": ("J",),
        "CCONJ": ("C",),
        "AUX": ("V",),
        "ADV": ("A",),
        "PART": ("O",),
        "NUM": ("J",),
        "CONJ": ("C",),
        "INTJ": ("I",),
        "PROPN": ("N",),
        "SCONJ": ("C",),
        "X": ("O",),
    }

    # =====================================================
    # INIT
    # =====================================================

    def __init__(
        self,
        dic_filepath: str | pathlib.Path | None,
    ) -> None:

        super().__init__(stop_word_removal_enabled=False)

        self._pickle_filepath = os.path.dirname(__file__) + "/mrc.pickle"

        has_pickle = os.path.exists(self._pickle_filepath)

        assert (
            isinstance(
                dic_filepath,
                (str, pathlib.Path),
            )
            and os.path.exists(dic_filepath)
        ) or has_pickle, ValueError(f"{dic_filepath} is invalid")

        # =================================================
        # FEATURES
        # =================================================

        self.features = np.array(
            self.__numerical_category_names_,
            dtype=object,
        )

        self._representation = np.array([])

        # =================================================
        # STAGGER
        # =================================================

        self._stagger = STagger()

        # =================================================
        # DICTIONARY
        # =================================================

        self._mrc_dic = {}

        if has_pickle:

            self.__load_dict_()

        else:

            self._dic_filepath = dic_filepath

            self.__read_dict_()

    # =====================================================
    # PICKLE SUPPORT
    # =====================================================

    def __getstate__(self):

        state = self.__dict__.copy()

        state.pop("_stagger", None)

        state.pop("_spacy_model", None)

        return state

    def __setstate__(self, state):

        self.__dict__.update(state)

        self._stagger = STagger()

    # =====================================================
    # HELPERS
    # =====================================================

    def __handle_tq2_(self, value: str) -> int:

        return int(value.upper() == "Q")

    def __convert_number_(self, value: str) -> int:

        try:

            return int(value, 10)

        except ValueError:

            return 0

    # =====================================================
    # LOAD PICKLE
    # =====================================================

    def __load_dict_(self) -> None:

        with open(
            self._pickle_filepath,
            "rb",
        ) as file:

            self._mrc_dic = pickle.load(file)

    # =====================================================
    # READ DICTIONARY
    # =====================================================

    def __read_dict_(self) -> None:

        with open(
            self._dic_filepath,
            "r+",
            encoding="utf-8",
        ) as file:

            self._mrc_dic = {
                word: value
                for line in file.readlines()
                for word, value in self.__process_dictionary_line_(line)
            }

        with open(
            self._pickle_filepath,
            "wb",
        ) as f:

            pickle.dump(
                self._mrc_dic,
                f,
            )

    # =====================================================
    # PROCESS LINE
    # =====================================================

    def __process_dictionary_line_(
        self,
        line: str,
    ) -> Generator[
        tuple[tuple[str, str], np.ndarray],
        None,
        None,
    ]:

        numerical_values = np.zeros(
            len(self.__numerical_category_names_),
            dtype=np.int32,
        )

        num_idx = 0

        # =================================================
        # NUMBERS
        # =================================================

        for num_idx, (
            start_rng,
            end_rng,
        ) in enumerate(self.__numerical_ranges_):

            numerical_values[num_idx] = self.__convert_number_(line[start_rng:end_rng])

        # =================================================
        # TQ2
        # =================================================

        num_idx += 1

        numerical_values[num_idx] = self.__handle_tq2_(line[43])

        # =================================================
        # CATEGORICAL
        # =================================================

        categorical_values = line[46:51]

        map_index_category_functions: list[Callable] = [
            self.__map_alphsyl_index_,
            self.__map_status_index_,
            self.__map_var_index_,
            self.__map_cap_index_,
            self.__map_irreg_index_,
        ]

        for map_fn, value in zip(
            map_index_category_functions,
            categorical_values,
        ):

            col_idx = map_fn(value)

            if col_idx != -1:

                numerical_values[col_idx] += 1

        # =================================================
        # TEXT
        # =================================================

        text_values = line[51:].split("|")

        word = text_values[0]

        wtype = remove_all_spaces(line[44].upper()) or str()

        yield (
            (word.lower(), wtype),
            numerical_values,
        )

    # =====================================================
    # MAPS
    # =====================================================

    def __map_alphsyl_index_(self, value: str):

        mapping = {
            "A": "ABBREVIATION",
            "S": "SUFFIX",
            "P": "PREFIX",
            "H": "HYPHENATED",
            "M": "MULTI-WORD",
        }

        final_value = value.upper()

        if final_value in mapping:

            return self.__numerical_category_names_.index(mapping[final_value])

        return -1

    def __map_status_index_(self, value: str):

        mapping = {
            "D": "DIALECT",
            "F": "ALIEN",
            "A": "ARCHAIC",
            "Q": "COLLOQUIAL",
            "C": "CAPITAL",
            "N": "ERRONEOUS",
            "E": "NONSENSE",
            "W": "NONCE WORD",
            "O": "OBSOLETE",
            "P": "POETICAL",
            "R": "RARE",
            "H": "RHETORICAL",
            "$": "SPECIALISED",
            "S": "STANDARD",
            "Z": "SUBSTANDARD",
        }

        final_value = value.upper()

        if final_value in mapping:

            return self.__numerical_category_names_.index(mapping[final_value])

        return -1

    def __map_var_index_(self, value: str):

        mapping = {
            "O": "PRONUNCIATION_DIFFER_STRESS",
            "B": "PRONUNCIATION_DIFFER",
        }

        final_value = value.upper()

        if final_value in mapping:

            return self.__numerical_category_names_.index(mapping[final_value])

        return -1

    def __map_cap_index_(self, value: str):

        if value.upper() == "C":

            return self.__numerical_category_names_.index("CAPITALIZATION")

        return -1

    def __map_irreg_index_(self, value: str):

        mapping = {
            "Z": "PLURAL",
            "Y": "SINGULAR",
            "B": "BOTH_SINGULAR_PLURAL",
            "N": "NO_PLURAL",
            "P": "PLURAL_ACT_SINGULAR",
        }

        final_value = value.upper()

        if final_value in mapping:

            return self.__numerical_category_names_.index(mapping[final_value])

        return -1

    # =====================================================
    # PREPROCESS
    # =====================================================

    def _pre_process_sentence(
        self,
        sentence: str,
    ) -> str:

        processed_sentence = super()._pre_process_sentence(sentence)

        return remove_punctuations(processed_sentence)

    # =====================================================
    # HELPER
    # =====================================================

    def __generate_representation_helper_(
        self,
        sentence: tuple[str, np.ndarray],
    ) -> np.ndarray:

        processed_sentence, tagged_sentence = sentence

        tokens = processed_sentence.split()

        row = np.zeros(
            len(self._features),
            dtype=np.int32,
        )

        for token, pos_tag in zip(
            tokens,
            tagged_sentence,
        ):

            wtypes = self.__universal_tag_to_wtype_.get(
                pos_tag,
                (),
            )

            for wtype in wtypes:

                mrc_value = self._mrc_dic.get(
                    (token, wtype),
                    None,
                )

                if mrc_value is not None:

                    row += mrc_value

        return row / max(1, len(tokens))

    # =====================================================
    # GENERATE
    # =====================================================

    def generate_representation(
        self,
        sentences: list[str],
        as_dataframe: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:

        self._representation = np.zeros(
            (
                len(sentences),
                len(self._features),
            ),
            dtype=np.float32,
        )

        processed_sentences = list(self.pre_process(sentences))

        tagged_sentences = list(self._stagger.run_tagging_pipe(processed_sentences))

        for idx, processed_sentence in enumerate(processed_sentences):

            self._representation[idx] = self.__generate_representation_helper_(
                (
                    processed_sentence,
                    tagged_sentences[idx],
                )
            )

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
# TEST
# =========================================================

if __name__ == "__main__":

    import time

    dic_path = (
        pathlib.Path(__file__).parents[3] / "shared/dictionaries/mrc" / "mrc2.dct"
    )

    mrc = MRC2(dic_filepath=dic_path)

    start = time.time()

    data = [
        "Use this metaclass to create an ABC.",
        "An ABC can be subclassed directly.",
        "You can also register unrelated classes.",
    ] * 10

    df = mrc.generate_representation(
        data,
        as_dataframe=True,
    )

    print(df)

    print(f"\nTempo: {time.time() - start:.2f}s")

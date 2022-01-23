from typing import Sequence, Dict, List, Tuple
from random import choice
from enum import Enum, auto
from abc import ABC, abstractmethod

NUM_ALPHABET = 5


class Result(Enum):
    Grey = auto()
    Yellow = auto()
    Green = auto()


class Attempt:
    guess: str
    results: List[Result]

    def __init__(self, guess: str, answer: str):
        self.guess = guess
        results = []
        for ab_guess, ab_ans in zip(guess, answer):
            result = Result.Grey
            if ab_guess == ab_ans:
                result = Result.Green
            elif ab_guess in answer:
                result = Result.Yellow
            results.append(result)
        self.results = results

    @property
    def is_success(self) -> bool:
        return all(result == Result.Green for result in self.results)

    def __repr__(self):
        m = {Result.Green: "🟩", Result.Grey: "⬜️", Result.Yellow: "🟨"}
        s = "".join(m[res] for res in self.results)
        return f"{self.guess} {s}"


class Game:
    answer: str
    attempts: List[Attempt]

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.attempts = []

    @classmethod
    def from_random(cls, word_list: List[str]):
        return cls(choice(word_list))

    def guess(self, word: str) -> List[Tuple[str, str]]:
        attempt = Attempt(word, self.answer)
        self.attempts.append(attempt)
        return attempt

    @property
    def is_ended(self) -> bool:
        if len(self.attempts) == 0:
            return False
        all_fail = len(self.attempts) == 6
        return self.attempts[-1].is_success or all_fail


def load_word_list() -> List[str]:
    # Ref: https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/a9e55d7e0c08100ce62133a1fa0d9c4f0f542f2c/wordle-answers-alphabetical.txt
    with open("./wordle-answers-alphabetical.txt") as f:
        words = [line.strip() for line in f.readlines()]
    return words


def build_freq_table(word_list: Sequence[str]) -> Dict[int, Dict[str, float]]:
    total_len = len(word_list)
    position = dict()
    for i in range(NUM_ALPHABET):
        freq = dict()
        for word in word_list:
            first_alphabet = word[i]
            freq[first_alphabet] = freq.get(first_alphabet, 0) + 1
        for k, v in freq.items():
            freq[k] = float(v) / total_len
        position[i] = freq
    return position


class Estimator(ABC):
    @abstractmethod
    def estimate(self) -> str:
        ...

    @abstractmethod
    def update(self) -> str:
        ...


class MaxLikelihood(Estimator):
    word_list: List[str]
    # position -> alphabet -> prob.
    freq_table: Dict[int, Dict[str, float]]

    def __init__(self, word_list: Sequence[str]):
        self.word_list = word_list
        self.freq_table = build_freq_table(word_list)

    def update(self, attpemt: Sequence[Tuple[str, str]]):
        word_list = self.word_list
        for i, (ab, res) in enumerate(zip(attpemt.guess, attpemt.results)):
            if res == Result.Green:
                word_list = [word for word in word_list if word[i] == ab]
            elif res == Result.Grey:
                word_list = [word for word in word_list if not (ab in word)]
            elif res == Result.Yellow:
                word_list = [
                    word for word in word_list if (ab in word) and word[i] != ab
                ]
            else:
                raise Exception("Unreachable")
        self.word_list = word_list
        self.freq_table = build_freq_table(word_list)

    @property
    def search_size(self):
        return len(self.word_list)

    def estimate(self):
        word_dict = dict()
        for word in self.word_list:
            prob = 1
            for i, s in enumerate(word):
                prob *= self.freq_table[i][s]
            word_dict[word] = prob
        top_words = sorted(word_dict, key=word_dict.get, reverse=True)
        top = top_words[0]

        return top, word_dict[top]


def play(game: Game, estimator: Estimator, verbose=False):
    if verbose:
        print("#    ", game.answer, "    ", estimator.search_size)
    while not game.is_ended:
        (guess, prob) = estimator.estimate()
        attempt = game.guess(guess)
        estimator.update(attempt)
        if verbose:
            print(attempt, estimator.search_size, "\t", prob)


def random_round():
    words = load_word_list()
    estimator = MaxLikelihood(words)
    game = Game.from_random(words)
    play(game, estimator, verbose=True)


def deterministic(questions: Sequence[str]):
    for answer in questions:
        words = load_word_list()
        estimator = MaxLikelihood(words)
        game = Game(answer)
        play(game, estimator, verbose=True)


def try_all_word_list():
    words = load_word_list()

    attempts_count = []
    result_count = []
    for word in words:
        game = Game(word)
        estimator = MaxLikelihood(words)
        play(game, estimator)
        attempts_count.append(len(game.attempts))
        result_count.append(game.attempts[-1].is_success)
    print("Total games", len(words))
    print("Avg attempts", sum(attempts_count) / len(attempts_count))
    print("Failing attempts", len(result_count) - sum(result_count))


if __name__ == "__main__":
    deterministic(["glade", "bilge", "rower", "viper", "blimp"])
    # for i in range(5):
    #     random_round()
    # try_all_word_list()


# Heuristics:
# - alphabet no repeat
# - include vowel as much as possible

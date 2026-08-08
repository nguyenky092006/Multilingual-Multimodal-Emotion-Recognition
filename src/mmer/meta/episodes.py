"""Deterministic N-way K-shot episodes over cached embedding datasets."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from mmer.data.cached import CachedEmbeddingDataset, CachedExample


@dataclass(frozen=True, slots=True)
class Episode:
    task_key: str
    classes: tuple[int, ...]
    support: tuple[CachedExample, ...]
    query: tuple[CachedExample, ...]

    def validate(self, n_way: int, k_shot: int, query_per_class: int) -> None:
        if len(self.classes) != n_way or len(set(self.classes)) != n_way:
            raise ValueError("episode class count does not match n_way")
        if len(self.support) != n_way * k_shot:
            raise ValueError("episode support size does not match N-way K-shot")
        if len(self.query) != n_way * query_per_class:
            raise ValueError("episode query size does not match N-way query count")
        support_ids = {item.sample_id for item in self.support}
        query_ids = {item.sample_id for item in self.query}
        if len(support_ids) != len(self.support) or len(query_ids) != len(self.query):
            raise ValueError("episode contains duplicate utterances")
        if support_ids & query_ids:
            raise ValueError("episode support/query utterances overlap")
        for label in self.classes:
            if sum(item.label == label for item in self.support) != k_shot:
                raise ValueError("episode support is not class balanced")
            if sum(item.label == label for item in self.query) != query_per_class:
                raise ValueError("episode query is not class balanced")

    @property
    def support_speakers(self) -> tuple[str, ...]:
        return tuple(sorted({item.speaker_id for item in self.support}))

    @property
    def query_speakers(self) -> tuple[str, ...]:
        return tuple(sorted({item.speaker_id for item in self.query}))


class EpisodeSampler:
    """Sample balanced episodes from one language/corpus task at a time."""

    TASK_FIELDS = {"global", "language", "corpus", "language_corpus", "speaker"}

    def __init__(
        self,
        dataset: CachedEmbeddingDataset,
        n_way: int,
        k_shot: int,
        query_per_class: int,
        episodes: int,
        seed: int,
        task_field: str = "corpus",
        disjoint_speakers: bool = False,
    ) -> None:
        if n_way <= 1 or k_shot <= 0 or query_per_class <= 0 or episodes <= 0:
            raise ValueError("episode dimensions and count must be positive; n_way must exceed one")
        if task_field not in self.TASK_FIELDS:
            raise ValueError(f"unsupported episodic task_field: {task_field}")
        self.dataset = dataset
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.query_per_class = int(query_per_class)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.task_field = task_field
        self.disjoint_speakers = bool(disjoint_speakers)
        self._tasks: dict[str, list[CachedExample]] = defaultdict(list)
        for example in dataset.examples:
            self._tasks[self._task_key(example)].append(example)
        self._eligible_tasks = [
            key for key, values in sorted(self._tasks.items()) if self._eligible(values)
        ]
        if not self._eligible_tasks:
            raise ValueError("no task can satisfy the requested N-way K-shot episode")

    def _task_key(self, example: CachedExample) -> str:
        if self.task_field == "global":
            return "global"
        if self.task_field == "language_corpus":
            return f"{example.language}|{example.corpus}"
        if self.task_field == "speaker":
            return example.speaker_id
        return str(getattr(example, self.task_field))

    def _eligible(self, examples: list[CachedExample]) -> bool:
        labels: dict[int, list[CachedExample]] = defaultdict(list)
        for item in examples:
            labels[item.label].append(item)
        if self.disjoint_speakers:
            speakers: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for item in examples:
                speakers[item.speaker_id][item.label] += 1
            for classes in itertools.combinations(sorted(labels), self.n_way):
                support_speakers = [
                    speaker
                    for speaker, counts in speakers.items()
                    if all(counts[label] >= self.k_shot for label in classes)
                ]
                query_speakers = [
                    speaker
                    for speaker, counts in speakers.items()
                    if all(counts[label] >= self.query_per_class for label in classes)
                ]
                if any(left != right for left in support_speakers for right in query_speakers):
                    return True
            return False
        return sum(
            len(values) >= self.k_shot + self.query_per_class for values in labels.values()
        ) >= self.n_way

    def _ordinary_episode(
        self, task_key: str, examples: list[CachedExample], rng: random.Random
    ) -> Episode | None:
        by_label: dict[int, list[CachedExample]] = defaultdict(list)
        for item in examples:
            by_label[item.label].append(item)
        candidates = [
            label
            for label, values in by_label.items()
            if len(values) >= self.k_shot + self.query_per_class
        ]
        if len(candidates) < self.n_way:
            return None
        classes = tuple(sorted(rng.sample(candidates, self.n_way)))
        support: list[CachedExample] = []
        query: list[CachedExample] = []
        for label in classes:
            selected = rng.sample(by_label[label], self.k_shot + self.query_per_class)
            support.extend(selected[: self.k_shot])
            query.extend(selected[self.k_shot :])
        episode = Episode(task_key, classes, tuple(support), tuple(query))
        episode.validate(self.n_way, self.k_shot, self.query_per_class)
        return episode

    def _speaker_disjoint_episode(
        self, task_key: str, examples: list[CachedExample], rng: random.Random
    ) -> Episode | None:
        by_speaker_label: dict[str, dict[int, list[CachedExample]]] = defaultdict(
            lambda: defaultdict(list)
        )
        labels: set[int] = set()
        for item in examples:
            by_speaker_label[item.speaker_id][item.label].append(item)
            labels.add(item.label)
        class_sets = list(itertools.combinations(sorted(labels), self.n_way))
        rng.shuffle(class_sets)
        speakers = sorted(by_speaker_label)
        for classes in class_sets:
            support_speakers = [
                speaker
                for speaker in speakers
                if all(len(by_speaker_label[speaker][label]) >= self.k_shot for label in classes)
            ]
            query_speakers = [
                speaker
                for speaker in speakers
                if all(
                    len(by_speaker_label[speaker][label]) >= self.query_per_class
                    for label in classes
                )
            ]
            pairs = [
                (left, right)
                for left in support_speakers
                for right in query_speakers
                if left != right
            ]
            if not pairs:
                continue
            support_speaker, query_speaker = rng.choice(pairs)
            support = [
                item
                for label in classes
                for item in rng.sample(by_speaker_label[support_speaker][label], self.k_shot)
            ]
            query = [
                item
                for label in classes
                for item in rng.sample(
                    by_speaker_label[query_speaker][label], self.query_per_class
                )
            ]
            episode = Episode(task_key, tuple(classes), tuple(support), tuple(query))
            episode.validate(self.n_way, self.k_shot, self.query_per_class)
            if set(episode.support_speakers) & set(episode.query_speakers):
                raise RuntimeError("speaker-disjoint episode construction failed")
            return episode
        return None

    def __iter__(self) -> Iterator[Episode]:
        rng = random.Random(self.seed)
        for _ in range(self.episodes):
            for _attempt in range(200):
                task_key = rng.choice(self._eligible_tasks)
                examples = self._tasks[task_key]
                episode = (
                    self._speaker_disjoint_episode(task_key, examples, rng)
                    if self.disjoint_speakers
                    else self._ordinary_episode(task_key, examples, rng)
                )
                if episode is not None:
                    yield episode
                    break
            else:
                raise RuntimeError("failed to construct an episode after 200 attempts")

    def audit(self) -> dict[str, object]:
        return {
            "task_field": self.task_field,
            "eligible_tasks": list(self._eligible_tasks),
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "query_per_class": self.query_per_class,
            "episodes": self.episodes,
            "disjoint_speakers": self.disjoint_speakers,
            "seed": self.seed,
        }

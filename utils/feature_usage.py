import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from collections import defaultdict


class FeatureExtractor:
    def __init__(self, get_audio, feature_family, transform=None):
        self.get_audio = get_audio
        self.transform = transform if transform is not None else lambda x: x
        self.feature_family = feature_family

        self.features = []
        self.feature_id = dict()
        for _, f_names in self.feature_family:
            for f in f_names:
                self.feature_id[f] = len(self.features)
                self.features.append(f)

    def feature_extraction(self, idx, row, cache):
        audio_key = (row[f"voice_{idx}"], row[f"L_{idx}"], row[f"R_{idx}"]) 
        values = []
        row_cache = cache[audio_key]
        audio = None

        for extractor, feature_names in self.feature_family:
            if sum(f in row_cache for f in feature_names) != len(feature_names):
                if audio is None:
                    audio = self.transform(self.get_audio(idx, row))
                feature_values = extractor.feature_extraction(audio)
                for key, value in feature_values.items():
                    row_cache[key] = value
                for f in feature_names:
                    values.append(feature_values[f])
            else:
                for f in feature_names:
                    values.append(row_cache[f])
        return torch.Tensor(values)

    def get_feature_names(self):
        features = []
        for _, f_names in self.feature_family:
            features += f_names
        return features


class SoundDataset(Dataset):
    def __init__(self, data, feature_exctractor, cache_path=None):
        self.data = data.reset_index(drop=True)
        self.cache_path = cache_path
        self.feature_exctractor = feature_exctractor
        self.feature_extraction = self.feature_exctractor.feature_extraction

        self._cache = dict()
        self._feature_cache = defaultdict(dict)
        self.cache_load()

        self.graph = defaultdict(lambda: defaultdict(list)) 
        for idx, row in self.data.iterrows():
            v1, v2 = row["author_first"], row["author_second"]
            self.graph[v1][v2].append(idx)
            self.graph[v2][v1].append(idx)

    def __len__(self):
        return self.data.shape[0] 

    def cache_load(self):
        if self.cache_path is not None and os.path.exists(self.cache_path):
            tmp = pd.read_csv(self.cache_path)
            for idx, row in tmp.iterrows():
                audio_key = (row["voice"], row["L"], row["R"])
                for key, value in row.items():
                    if pd.isnull(value) or key in ["voice", "L", "R"]:
                        continue
                    self._feature_cache[audio_key][key] = value

    def cache_save(self):
        if self.cache_path is not None and len(self._feature_cache) != 0:
            columns = set()
            for row in self._feature_cache.values():
                columns.update(row.keys())

            data_rows = []
            for audio_key, row in self._feature_cache.items():
                cur = list(audio_key)
                for column in columns:
                    cur.append(row[column] if column in row else None)
                data_rows.append(cur)

            columns = ["voice", "L", "R"] + list(columns)
            tmp = pd.DataFrame(data=data_rows, columns=columns)
            tmp.to_csv(self.cache_path, index=False)

    def __getitem__(self, i):
        value = []
        row = self.data.loc[i]
        for idx in ["first", "second"]:
            audio_key = (row[f"voice_{idx}"], row[f"L_{idx}"], row[f"R_{idx}"]) 
            if audio_key not in self._cache:
                self._cache[audio_key] = self.feature_extraction(
                    idx, row, self._feature_cache
                )
            value.append(self._cache[audio_key])
        return tuple(value)

    def generator(self):
        def gen():
            authors = list(self.graph.keys())
            ans = []
            change = 10
            while change != 0:
                change -= 1
                bad_authors = []

                np.random.shuffle(authors)
                for i in range(1, len(authors), 2):
                    if authors[i - 1] in self.graph[authors[i]]:
                        yield self[np.random.choice(self.graph[authors[i]][authors[i - 1]])]
                    else:
                        bad_authors.append(authors[i - 1])
                        bad_authors.append(authors[i])
                authors = bad_authors

        return gen()
                


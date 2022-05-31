import pandas as pd


class Evaluator:
    def __init__(self, dataset, info):
        self.dataset = dataset
        self.info = info
        self.info_mean = self.info[["author", "rating"]].groupby(["author"]).mean()
        self.rating = self.info.groupby(["author"])

    def evaluate(self, predict):
        values = []
        cache = dict()
        for idx, row in self.info.iterrows():
            f = (row["voice"], row["L"], row["R"])
            if f not in cache:
                cache[f] = predict(self.dataset.get_value(f))
            values.append([row["author"], cache[f]])
        self.values = pd.DataFrame(values, columns=["author", "estimation"])
        self.values = self.values.groupby(["author"]).mean().reset_index()
        self.values = self.values.join(self.info_mean, on="author", how="left")

    def kendall_coefficients(self):
        discordant = 0

        rank = self.values["rating"]
        estimation = self.values["estimation"]
        for i in range(len(rank)):
            for j in range(i):
                if rank[i] == rank[j]:
                    continue
                if estimation[i] == estimation[j]:
                    discordant += 1
                elif (rank[i] < rank[j]) ^ (estimation[i] < estimation[j]):
                    discordant += 1
        
        total = len(rank) * (len(rank) - 1) // 2
        concordant = total - discordant
        return (concordant - discordant) / total

import pandas as pd


class Evaluator:
    def __init__(self, dataset, info):
        self.dataset = dataset
        self.info = pd.read_csv(info)[["author", "raiting"]].groupby(["author"]).mean()
        self.raiting = self.info.groupby(["author"])

    def evaluate(self, predict):
        values = []
        cache = dict()
        for i in range(len(self.dataset)):
            row = self.dataset.data.loc[i]
            f = self.dataset[i]
            if f[0] not in cache:
                cache[f[0]] = predict(f[0])
            if f[1] not in cache:
                cache[f[1]] = predict(f[1])
            v = [cache[f[0]], cache[f[1]]]
            values.append([row["author_first"], v[0]])
            values.append([row["author_second"], v[1]])
        self.values = pd.DataFrame(values, columns=["author", "estimation"])
        self.values = self.values.groupby(["author"]).mean().reset_index()
        self.values = self.values.join(self.info, on="author", how="left")

    def kendall_coefficients(self):
        discordant = 0

        rank = self.values["raiting"]
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

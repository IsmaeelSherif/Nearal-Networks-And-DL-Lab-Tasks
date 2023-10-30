


class MinMaxNormalizer():
    def fit(self, dfColumn):
        self.min = dfColumn.min()
        self.max = dfColumn.max()
    
    def transform(self, dfColumn):
        min = self.min
        max = self.max
        normalized_df = (dfColumn - min)/(max - min)
        return normalized_df
    
    def fit_transform(self, dfColumn):
        self.fit(dfColumn)

        min = self.min
        max = self.max
        normalized_df = (dfColumn - min)/(max - min)
        return normalized_df
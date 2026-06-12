import numpy as np
import math

class PresenceAbundanceFilter(object):
    def __init__(self, th_presence, th_abundance):
        self.th_presence = th_presence
        self.th_abundance = th_abundance
        self.fitted = False
        self.count_presence = 0
        self.count_abundance = 0
        self.removed_columns = []

    def fit(self, df):
        """
            1. Keep just the features present in at least th_presence=10 samples
            2. Keep only the features whose max values (among all the samples) is greater than th_presence=0.1/0.01
            3. Apply the Filtering within each class
            4. Use the OR logic after the filtering by class: if a feature is present in a class but not in the other, I keep the feature for both of them
        """
        df_zeros = (df != 0) * 1

        columns_to_remove_presence = [col for col in df_zeros.columns if
                                      (df_zeros[col] != 0).sum() < len(df_zeros) * self.th_presence]

        count_presence = df.shape[1] - len(columns_to_remove_presence)
        for col in columns_to_remove_presence:
            df[col] = np.nan

        columns_to_remove_abundance = [col for col in df_zeros.columns if (max(df[col]) < self.th_abundance)]
        # print(columns_to_remove_presence)

        count_abundance = df.shape[1] - len(columns_to_remove_abundance) - len(columns_to_remove_presence)
        for col in columns_to_remove_abundance:
            df[col] = np.nan
        # Remove all the columns containing NaN values
        removed_columns = [col for col in df_zeros.columns if math.isnan(max(df[col]))]
        self.count_presence = count_presence
        self.count_abundance = count_abundance
        self.removed_columns = np.array(removed_columns)
        self.fitted = True

    def __call__(self, df):
        if not self.fitted:
            raise RuntimeError("Fit first before applying filter")
        return df.drop(columns=self.removed_columns)



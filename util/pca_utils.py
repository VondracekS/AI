import pandas as pd
from sklearn.PCA import PCA


def get_pca(df, target, pca_var_threshold=.90, vars_untransformed=[]):

    pca_t = PCA(n_components=pca_var_threshold).fit_transform(
        df.drop([target]+vars_untransformed, axis=1))
    pca_t = pd.DataFrame(pca_t, index=df.index, columns=[f"pc_{i+1}" for i in range(pca_t.shape[1])])

    return pca_t

def add_pca(df, target, pca_var_threshold=.90, vars_untransformed=[]):

    df_pca_added = df.copy()
    df_pca_added = pd.concat([get_pca(df, target, pca_var_threshold, vars_untransformed), df,
                              df_pca_added],
                             axis=1)
    return df_pca_added
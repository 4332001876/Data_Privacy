from sklearn import preprocessing


def scale(dataset):
    raw_dataset = dataset.get_dataset() # 获取数据集

    start_col = 2 if dataset.has_label else 1 # 从第2列开始是feature
    scaled_feats = preprocessing.scale(raw_dataset[:, start_col:], copy=False)
    raw_dataset[:, start_col:] = scaled_feats # 对feature进行标准化

    dataset.set_dataset(raw_dataset) # 更新数据集

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import re
from nltk.tokenize import word_tokenize
import joblib
import pickle


def func(ser):
    nans = np.count_nonzero(pd.isnull(ser))
    dist_val = len(pd.unique(ser.dropna()))
    total_val = ser.shape[0]
    mean = 0
    std_dev = 0
    min_val = 0
    max_val = 0
    if is_numeric_dtype(ser):
        mean = np.mean(ser)

        if pd.isnull(mean):
            mean = 0
            std_dev = 0
            min_val = 0
            max_val = 0
        else:
            std_dev = np.std(ser)
            min_val = float(np.min(ser))
            max_val = float(np.max(ser))

    ratio_dist_val = dist_val * 100.0 / total_val
    ratio_nans = nans * 100.0 / total_val

    summary_stats = [total_val, nans, dist_val, mean, std_dev, min_val, max_val, ratio_dist_val, ratio_nans]

    unique_vals = np.random.choice(pd.unique(ser), 5)

    res = summary_stats + list(unique_vals)
    return pd.Series(res)


def process_sample(ser):
    from nltk.corpus import stopwords

    del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
    del_reg = re.compile(del_pattern)

    delimeters = r"(,|;|\|)"
    delimeters = re.compile(delimeters)

    url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    url_reg = re.compile(url_pat)

    email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
    email_reg = re.compile(email_pat)

    stop_words = set(stopwords.words('english'))

    curlst = [ser['sample_1'], ser['sample_2'], ser['sample_3'], ser['sample_4'], ser['sample_5']]

    delim_cnt, url_cnt, email_cnt, date_cnt = 0, 0, 0, 0
    chars_totals, word_totals, stopwords, whitespaces, delims_count = [], [], [], [], []

    for value in curlst:
        word_totals.append(len(str(value).split(' ')))
        chars_totals.append(len(str(value)))
        whitespaces.append(str(value).count(' '))

        if del_reg.match(str(value)):  delim_cnt += 1
        if url_reg.match(str(value)):  url_cnt += 1
        if email_reg.match(str(value)):  email_cnt += 1

        delims_count.append(len(delimeters.findall(str(value))))

        tokenized = word_tokenize(str(value))
        stopwords.append(len([w for w in tokenized if w in stop_words]))

        try:
            _ = pd.Timestamp(value)
            date_cnt += 1
        except ValueError:
            date_cnt += 0

    d = {'has_delimiters': True if delim_cnt > 2 else False,
         'has_url': True if url_cnt > 2 else False,
         'has_email': True if email_cnt > 2 else False,
         'has_date': True if date_cnt > 2 else False,
         'mean_word_count': np.mean(word_totals),
         'std_dev_word_count': np.std(word_totals),
         'mean_stopword_total': np.mean(stopwords),
         'stdev_stopword_total': np.std(stopwords),
         'mean_char_count': np.mean(chars_totals),
         'stdev_char_count': np.std(chars_totals),
         'mean_whitespace_count': np.mean(whitespaces),
         'stdev_whitespace_count': np.std(whitespaces),
         'mean_delim_count': np.mean(delims_count),
         'stdev_delim_count': np.std(delims_count),
         }

    d['is_list'] = True if d['has_delimiters'] is True and d['mean_char_count'] < 100 else False
    d['is_long_sentence'] = True if d['mean_word_count'] > 10 else False

    return ser.append(pd.Series(d))


def feature_extraction(data):
    data1 = data[
        ['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val',
         'has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
         'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
         'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
         'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
         'is_list', 'is_long_sentence']].copy()

    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    arr = [str(x) for x in data['Attribute_name']]

    vectorizerName = joblib.load("resources/dictionaryName.pkl")

    X = vectorizerName.transform(arr)

    attr_df = pd.DataFrame(X.toarray())

    data2 = pd.concat([data1, attr_df], axis=1, sort=False)

    return data2


def load_rf(df, model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    res = model.predict(df).tolist()
    return res


def infer_data(df, model_path, sample=1):

    df = df.sample(frac=sample)

    new_df = df.apply(func)

    column_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
                    'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']

    new_df = new_df.T.reset_index()

    new_df.columns = column_names

    d = new_df.apply(process_sample, axis=1)

    data2 = feature_extraction(d)

    res = load_rf(data2, model_path)

    labels_mapping = {
        0: "numeric",
        1: "categorical",
        2: "datetime",
        3: "sentence",
        4: "url",
        5: "embedded_number",
        6: "list",
        7: "not-generalizable",
        8: "custom-specific"
    }

    columns_datatypes = {col[0]:labels_mapping[col[1]] for col in zip(df.columns, res)}

    return columns_datatypes
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

def resumetable(df):
    """
    Return summary sof the dataframe e.g. Missing values, unique values, first few values of the dataset

    :param
        df:  pandas dataframe
    :return:
        summary: dataframe that contains summary of  df
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    return summary


def group_email_domain(df):
    """
    Function that group  email domain bty the service provider  grouped  email domain that
    has very few coiunts into  category 'other'
    :param df:  pandas dataframe
    :return:  df:  pandas dataframe
    """
    for email in ['P_emaildomain', 'R_emaildomain']:
        df.loc[df[email].
                   isin(['gmail.com', 'gmail']), email] = 'Google Mail'
        df.loc[df[email].
                   isin(['yahoo.com', 'ymail.com', 'yahoo.com.mx',
                         'yahoo.co.jp', 'yahoo.fr', 'yahoo.co.uk',
                         'yahoo.es', 'yahoo.de']), email] = 'Yahoo Mail'
        df.loc[df[email].
                   isin(['hotmail.com', 'outlook.com', 'msn.com',
                         'live.com', 'live.com.mx', 'outlook.es',
                         'hotmail.fr', 'hotmail.co.uk', 'live.fr',
                         'hotmail.es', 'hotmail.de']), email] = 'Microsoft mail'
        df.loc[df[email].
                   isin(['icloud.com', 'me.com', 'mac.com']), email] = 'Apple mail'

        df.loc[df[email].
                   isin(df[email].
                        value_counts()[df[email].
                        value_counts() <= 1000].index), email] = 'Others'
    return df


def plot_cat_features(df, col, lim=2000):
    """
    Extension of  count_fraud_plot  with addition of  box plot
    :param df:  pandas dataframe
    :param col:  categorical colnum
    :param lim: int that limit the transaction amt
    :return:
    """

    plt.figure(figsize=(14, 10))
    plt.subplot(221)
    g = sns.countplot(x=col, data=df)
    g.set_title(col + ' Distribution', fontsize=14)
    # g.set_xlabel(col + ' Name', fontsize=10)
    g.set_ylabel('Count', fontsize=14)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x() + p.get_width() / 2.,
               height,
               f'{height / df.shape[0] * 100:.2f}%',
               ha='center', fontsize=10)

    plt.subplot(222)
    g1 = df.groupby(col)['isFraud'].mean() \
        .sort_index() \
        .plot(kind='bar',
              title='Percentage of Fraud by ' + col,
              color=sns.color_palette())
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x() + p.get_width() / 2.,
                height,
                f'{height * 100:.2f}%',
                ha='center', fontsize=10)

    plt.subplot(212)
    g2 = sns.boxenplot(x=col, y='TransactionAmt', hue='isFraud',
                       data=df[df['TransactionAmt'] <= lim])
    g2.set_title(f"{col} boxplot by TransactionAmt and Fraud")
    g2.set_ylabel("Transaction Values in $")

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Function to reduce memory usage of dataframe
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_and_merge(raw_data_path, split, merge=True):
    """
    Function that load the dataset and merge the dataset if the argument merge is True

    :param raw_data_path: str. the folder name that contain the data
    :param split:  str. train / test
    :param merge: bool. if True merge transaction and identity dataset
    :return: merge_df  or  transaction and identity dataset
    """

    transaction = pd.read_csv(f"{raw_data_path}/{split}_transaction.csv")
    identity = pd.read_csv(f"{raw_data_path}/{split}_identity.csv")

    if merge:
        merge_df = transaction.merge(identity, how='left', left_index=True, right_index=True)
        return merge_df
    return transaction, identity


def high_duplicates_col(df, base_columns):
    """
    Return a list  of column names that  have high similarities to c1 column
    :param df: pandas dataframe
    :param base_columns:  columns from original data
    :return:  list of columns names that have high similarities  to other columns
    """
    duplicates = []
    i = 0
    for c1 in base_columns:
        i += 1
        for c2 in base_columns[i:]:
            if c1 != c2:
                if (np.sum((df[c1].values == \
                            df[c2].values).astype(int)) / len(df)) > 0.95:
                    duplicates.append(c2)

    return list(set(duplicates))


def get_cols_to_drop(df, base_columns):
    """
    Return a list of columns that have many null values, high similarity columns and columns that dominated by one value
    :param df:  pandas dataframe
    :param base_columns:  list. columns from original data
    :return:  cols_to_drop
    """
    many_null_cols = [col for col in base_columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    big_top_value_cols = [col for col in base_columns if
                          df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    cols_to_drop = list(set(many_null_cols + big_top_value_cols + high_duplicates_col(df, base_columns)))

    if 'isFraud' in cols_to_drop:
        cols_to_drop.remove('isFraud')

    return cols_to_drop


def make_day_feature(df, timecol='TransactionDT'):
    """
    Create  day feature , encoded as 0-6.
    :param df: pandas DaraFrame
    :param timecol:  str. Name of  time column in df
    :return:  encoded_days  int. 0-6
    """
    days = df[timecol] / (3600 * 24)
    encoded_days = np.floor(days - 1) % 7
    return encoded_days


def make_hour_feature(df, timecol='TransactionDT'):
    """
    Create hour feature, encoded as 0-23.
    :param df:  pandas DataFrame
    :param timecol:  str. Name of time column in df
    :return:  encoded_hours. int. 0-23
    """
    hours = df[timecol] / (3600)
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


impt_features = ['isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',
                 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4',
                 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                 'V12', 'V13', 'V20', 'V24', 'V36', 'V37', 'V38', 'V44', 'V45', 'V53', 'V54', 'V55', 'V56', 'V61', 'V62', 'V64',
                 'V67', 'V75', 'V76', 'V77', 'V78', 'V79', 'V82', 'V83', 'V86', 'V87', 'V102', 'V130', 'V131', 'V133', 'V139',
                 'V140', 'V143', 'V149', 'V152', 'V156', 'V165', 'V169', 'V171', 'V189', 'V201', 'V203', 'V208', 'V245', 'V251',
                 'V258', 'V261', 'V262', 'V264', 'V266', 'V267', 'V270', 'V277', 'V281', 'V282', 'V283', 'V285', 'V289', 'V291',
                 'V294', 'V296', 'V307', 'V308', 'V309', 'V310', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V320',
                 'V323', 'V336', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11',
                 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
                 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                 'DeviceType', 'DeviceInfo']




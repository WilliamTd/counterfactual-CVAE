"""
part of the code is from https://physionet.org/content/ptb-xl/1.0.1/#files-panel
"""
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def main():
    path = 'raw_data/'
    ctype = "superdiagnostic"
    output_path = "processed"
    sampling_rate=100
    print('loading . . .')
    
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    ecg_id = np.load('ecg_id.npy')
    Y = Y[Y.index.isin(ecg_id)]
    X = X[ecg_id-1]
    X = np.transpose(X, (0,2,1))

    mlb = MultiLabelBinarizer()
    target_y = mlb.fit_transform(Y.diagnostic_superclass)
    print(X.shape)
    print(Y.shape)
    print(target_y.shape)
    
    # save
    np.savez_compressed(f"{output_path}/compressed_X.npz", a=X)
    Y.to_csv(f"{output_path}/Y.csv", index=False)
    np.save(f"{output_path}/y_{ctype}.npy", target_y)
    pickle.dump(mlb, open(f"{output_path}/mlb_{ctype}.pkl", 'wb'))
    
    print('Done')

if __name__ == '__main__':

    main()
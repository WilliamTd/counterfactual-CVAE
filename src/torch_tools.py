import numpy as np
import pickle
import torch
import json
import torch
import random

def set_seeds(seed, device):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python
    if device == "gpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def load_model_class(model_path, ClassModel):
    state_dict = pickle.load(open(f"{model_path}/state_dict.pkl", "rb" ))
    model_params = json.load(open(f"{model_path}/model_params.json","rb"))
    scaler = pickle.load(open(f"{model_path}/scaler.pkl", "rb" ))
    model = ClassModel(scaler=scaler, **model_params)
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model.train_mse = np.load(f"{model_path}/train_mse.npy")
    model.train_loss = np.load(f"{model_path}/train_loss.npy")
    model.train_kl = np.load(f"{model_path}/train_kl.npy")
    model.test_kl = np.load(f"{model_path}/test_kl.npy")
    model.test_loss = np.load(f"{model_path}/test_loss.npy")
    model.test_mse = np.load(f"{model_path}/test_mse.npy")
    model.kl_scaler = np.load(f"{model_path}/kl_scaler.npy")
    return model

class DatasetTest(torch.utils.data.Dataset):
    """
    x : ecg numpy.array(n_sample, n_lead, length_ts)
    target_y : numpy array ;one hot encoded target
    ecg_id : int ecg id
    output_size : output_ts_length
    """
    def __init__(self, x, target_y, ecg_id, output_size):
        assert x.shape[0] == target_y.shape[0] 
        assert x.shape[0] == ecg_id.shape[0]
        self.x = torch.tensor(x.astype(np.float32))
        self.target_y = target_y.astype(np.float32)
        self.ecg_id = ecg_id.astype(np.float32)
        self.input_size = x.shape[2]
        self.output_size = output_size
        self.nb_variables = x.shape[1]

        self.start_indices = np.arange(
            0, self.input_size - self.output_size + 1, self.output_size // 2
        ).tolist()
        if (
            self.input_size
            - (self.start_indices[-1] + self.output_size) / self.output_size
            > 0.33
        ):
            self.start_indices += [self.input_size - self.output_size]
        self.start_indices = np.asarray(self.start_indices)
        
        self.nb_crop = self.start_indices.shape[0]
    def __getitem__(self, index):
        # overlapp crop
        x_ts_all = torch.empty(
            (self.start_indices.shape[0], self.nb_variables, self.output_size)
        )
        for i, start_index in enumerate(self.start_indices):
            x_ts_all[i, :, :] = self.x[
                index, :, start_index : (start_index + self.output_size)
            ]

        return x_ts_all, self.target_y[index], self.ecg_id[index]

    def __len__(self):
        return self.x.shape[0]
    
    
class DatasetSized(torch.utils.data.Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
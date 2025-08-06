import pickle
from utils.utils import MolTransformFn, MolCollateFn
import os
from utils.inmemory_dataset import InMemoryDataset

def get_data_loader(args, mode):
    collate_fn = MolCollateFn(args)
    transform_fn = MolTransformFn(args)

    if mode == 'train':
        train_data_path = os.path.join(args.data_dir_path, 'train.pkl')
        valid_data_path = os.path.join(args.data_dir_path, 'valid.pkl')
        train_data_list = pickle.load(open(train_data_path, 'rb'))
        val_data_list = pickle.load(open(valid_data_path, 'rb'))

        train_data, valid_data = InMemoryDataset(train_data_list), InMemoryDataset(val_data_list)

        train_data.transform(transform_fn, num_workers=args.num_workers)
        valid_data.transform(transform_fn, num_workers=args.num_workers)
            
        train_dl = train_data.get_data_loader(batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid_data.get_data_loader(batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        return train_dl, valid_dl

    elif mode in ['test', 'test_xood', 'test_yood', 'test_xyood']:
        test_path = os.path.join(args.data_dir_path, mode + '.pkl')
        test_list = pickle.load(open(test_path, 'rb'))

        test_data = InMemoryDataset(test_list)

        test_data.transform(transform_fn, num_workers=args.num_workers)
            
        test_dl = test_data.get_data_loader(batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        return test_dl
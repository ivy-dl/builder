# global
import os
import cv2
import ivy
import json
import logging
import numpy as np
import multiprocessing
from ivy_builder.dataset import MapDataset
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.data_loaders.specs.json_data_loader_spec import JSONDataLoaderSpec


# noinspection PyUnresolvedReferences
class JSONDataLoader(DataLoader):

    def __init__(self, data_loader_spec: JSONDataLoaderSpec):
        super(JSONDataLoader, self).__init__(data_loader_spec)

        # cpus
        if 'num_workers' in data_loader_spec:
            self._total_num_workers = data_loader_spec.num_workers
        else:
            self._total_num_workers = multiprocessing.cpu_count()

        # first frame validity
        if 'first_frame_validity_fn' in data_loader_spec:
            self._first_frame_validity_fn = data_loader_spec.first_frame_validity_fn
        else:
            self._first_frame_validity_fn = None

        # data loader specification
        self._spec = data_loader_spec
        self._container_data_dir = os.path.join(self._spec.dataset_spec.dirs.dataset_dir, 'containers/')

        # variables
        self._window_size = self._spec.window_size
        if 'sequence_lengths' in self._spec.dataset_spec:
            self._fixed_sequence_length = isinstance(self._spec.dataset_spec.sequence_lengths, int)
            if self._fixed_sequence_length:
                self._windows_per_seq = self._spec.dataset_spec.sequence_lengths - (self._window_size - 1)
            else:
                self._windows_per_seq = ivy.array(self._spec.dataset_spec.sequence_lengths) - (self._window_size - 1)
        else:
            self._fixed_sequence_length = False
        self._batch_size = self._spec.batch_size

        # train and validation idxs
        start_idx_train = 0
        end_idx_train = self._spec.num_training_sequences - 1
        start_idx_valid = self._spec.num_training_sequences
        end_idx_valid = self._spec.num_sequences_to_use - 1

        # compute num workers for each component
        self._compute_num_workers()

        # train dataset
        self._training_dataset = self._get_map_dataset(start_idx_train, end_idx_train)
        # ToDo: remove with_prefetching=... debug code below
        self._training_iterator = self._training_dataset.to_iterator(
            'to_iterator', with_prefetching=False, parallel_method=self._spec.prefetch_method,
            to_gpu=self._spec.prefetch_to_gpu, ivyh=ivy)

        # validation
        if self._spec.num_training_sequences < self._spec.num_sequences_to_use:
            self._validation_dataset = self._get_map_dataset(start_idx_valid, end_idx_valid)
            # ToDo: remove with_prefetching=... debug code below
            self._validation_iterator = self._validation_dataset.to_iterator(
                'to_iterator', with_prefetching=False, parallel_method=self._spec.prefetch_method,
                to_gpu=self._spec.prefetch_to_gpu, ivyh=ivy)
        else:
            self._validation_dataset = None
            self._validation_iterator = None

        # dummy batch
        self._dummy_batch = None

        # counter
        self._counter = 0

    # Built-Ins #
    # ----------#

    def __del__(self):
        self.close()

    # Dataset in RAM #
    # ---------------#

    @staticmethod
    def _to_tensor(x, ivyh=None):
        ivyh = ivy.default(ivyh, ivy)
        if type(x) == str:
            x = [[list(x.encode())]]
            return ivyh.array(x, dtype_str='uint8')
        return ivyh.array(x, dtype_str='float32')

    @staticmethod
    def _load_container_filepaths_as_lists(cont_dir, starting_example, ending_example):

        if not os.path.isdir(cont_dir):
            raise Exception('container dir ' + cont_dir + ' does not exist')

        cont_filenames = os.listdir(cont_dir)
        cont_filenames.sort()
        cont_paths = [os.path.join(cont_dir, cont_filename) for cont_filename in cont_filenames]

        grouped_conts = []
        group = []
        cont_num = 0
        for i, cont_path in enumerate(cont_paths):
            seq_id = int(cont_path.split('/')[-1].split('_')[0])
            if seq_id < starting_example:
                continue
            elif seq_id > ending_example:
                break
            trimmed_string = cont_path.split('/')[-1].split('_')[-2]
            current_cont_num = int(trimmed_string) if str.isdigit(trimmed_string) else int(trimmed_string[1:])
            if current_cont_num != cont_num and seq_id != starting_example:
                cont_num = current_cont_num
                grouped_conts.append(group)
                group = []
            group.append(cont_path)
        grouped_conts.append(group)
        return grouped_conts

    def _get_containers_w_filepath_img_entries_as_tensor_slices(self, container_filepaths):
        # noinspection PyUnusedLocal
        all_containers = list()
        logging.info('loading containers into RAM...')
        num_seqs = len(container_filepaths)
        max_seq_len = max(max(len(item) for item in container_filepaths), self._window_size)
        for seq_idx, seq in enumerate(container_filepaths):
            if seq_idx % 10000 == 0:
                logging.info('sequence {} of {}'.format(seq_idx, num_seqs))
            window_containers = list()
            container = None
            seq_len = 0
            for seq_len, filepath in enumerate(seq):
                if filepath == '':
                    seq_len -= 1
                    break
                with open(filepath) as fp:
                    container_dict = json.load(fp)
                container = ivy.Container(container_dict).map(lambda x, kc: self._to_tensor(x))
                window_containers.append(container)
            window_containers += [container] * (max_seq_len - seq_len - 1)  # padding for shorter sequences
            joined_window_containers = ivy.Container.concat(window_containers, 1)
            all_containers.append(joined_window_containers)
        return ivy.Container.concat(all_containers, 0)

    # Multiprocessing Sizes #
    # ----------------------#

    def _compute_num_workers(self):

        # init
        num_workers = self._total_num_workers
        self._num_workers = ivy.Container()

        # post processed
        self._num_workers.post_processed = 1

        # from numpy
        self._num_workers.from_numpy = 1

        # batched
        self._num_workers.batched = 1

        # loaded data
        self._num_workers.loaded_data = min(num_workers, self._batch_size)

        # ToDo: add multi-processing support for these lower level datasets

        # shuffled
        self._num_workers.shuffled = 1

        # unbatch
        self._num_workers.unbatched = 1

        # windowed
        self._num_workers.windowed = 1

        # valid first frames
        self._num_workers.valid_first_frames = 1

        # keychain pruned
        self._num_workers.keychain_pruned = 1

        # parsed json
        self._num_workers.parsed_json = 1

        # loaded json
        self._num_workers.loaded_json = 1

    # Dynamic Windowing #
    # ------------------#

    def _group_tensor_into_windowed_tensor_simple(self, x, seq_info, ivyh):
        if self._fixed_sequence_length:
            return ivyh.reshape(ivyh.gather_nd(x, ivyh.array(self._gather_idxs)),
                                (self._windows_per_seq, self._window_size) + x.shape[1:])
        else:
            num_windows_in_seq = int(ivyh.to_numpy(ivyh.maximum(seq_info.length[0] - self._window_size + 1, 1)))
            window_idxs_in_seq = ivyh.arange(num_windows_in_seq, 0, 1)
            gather_idxs = ivyh.tile(ivyh.reshape(ivyh.arange(self._window_size, 0, 1), (1, self._window_size)),
                                   (num_windows_in_seq, 1)) + ivyh.expand_dims(window_idxs_in_seq, -1)
            gather_idxs_flat = ivyh.reshape(gather_idxs, (self._window_size * num_windows_in_seq, 1))
            return ivyh.reshape(ivyh.gather_nd(x, gather_idxs_flat),
                               (num_windows_in_seq, self._window_size) + x.shape[1:])

    def _group_tensor_into_windowed_tensor(self, x, valid_first_frame, ivyh):
        if self._window_size == 1:
            valid_first_frame_pruned = ivyh.cast(valid_first_frame[:, 0], 'bool')
        else:
            valid_first_frame_pruned = ivyh.cast(valid_first_frame[:1-self._window_size, 0], 'bool')
        if ivyh.reduce_sum(ivyh.cast(valid_first_frame_pruned, 'int32'))[0] == 0:
            valid_first_frame_pruned =\
                ivyh.cast(ivyh.one_hot(0, self._sequence_lengths[0] - self._window_size + 1), 'bool')
        window_idxs_single = ivyh.indices_where(valid_first_frame_pruned)

        gather_idxs_list = list()
        for w_idx in window_idxs_single:
            gather_idxs_list.append(ivyh.expand_dims(ivyh.arange(w_idx[0] + self._window_size, w_idx[0], 1), 0))
        gather_idxs = ivyh.concatenate(gather_idxs_list, 0)
        gather_idxs = ivyh.reshape(gather_idxs, (-1, 1))
        num_valid_windows_for_seq = ivyh.shape(window_idxs_single)[0:1]
        return ivyh.reshape(ivyh.gather_nd(x, gather_idxs),
                           ivyh.concatenate((num_valid_windows_for_seq,
                                            ivyh.array([self._window_size]), ivyh.shape(x)[1:]), 0))

    def _group_container_into_windowed_container(self, container, ivyh):
        ivyh = ivy.default(ivyh, ivy)
        if self._first_frame_validity_fn is not None:
            return container.map(
                lambda x, _: self._group_tensor_into_windowed_tensor(x, container.valid_first_frame, ivyh))
        else:
            if 'seq_info' in container:
                seq_info = container.seq_info
            else:
                seq_info = None
            return container.map(
                lambda x, _: self._group_tensor_into_windowed_tensor_simple(x, seq_info, ivyh))

    # Dynamic File Reading #
    # ---------------------#

    # json to container

    @staticmethod
    def _load_json_files(containers, ivyh):
        ivyh = ivy.default(ivyh, ivy)
        read_files = list()
        for j_fpath in containers.fpaths:
            if j_fpath != '':
                with open(j_fpath, 'r') as file:
                    read_str = file.read()
            else:
                read_str = ''
            read_files.append(read_str)
        return ivy.Container({'json_str': read_files}, ivyh=ivyh)

    def _parse_json_strings(self, containers, ivyh):
        ivyh = ivy.default(ivyh, ivy)
        json_strings_stack = containers.json_str
        highest_idx_entry = len([item for item in containers.json_str if item != '']) - 1
        json_container_stack = [ivy.Container(json.loads(json_str), ivyh=ivyh).map(
            lambda x, kc: self._to_tensor(x, ivyh))[0] if json_str != '' else
                                ivy.Container(json.loads(json_strings_stack[highest_idx_entry]), ivyh=ivyh).map(
                                    lambda x, kc: self._to_tensor(x, ivyh))[0] for json_str in json_strings_stack]
        return ivy.Container.concat(json_container_stack, 0, ivyh=ivyh)

    # container pruning

    def _prune_unused_key_chains(self, container, ivyh=None):
        for unused_key_chain in self._spec.unused_key_chains:
            container = container.prune_key_chain(unused_key_chain)
        return container

    # arrays

    def _array_fn(self, filepaths_in_window, ivyh):
        conts = list()
        # ToDo: replace this with map_fn function once implemented
        for filepath in filepaths_in_window:
            str_path = bytearray(ivyh.to_numpy(filepath).tolist()).decode()
            full_path = os.path.abspath(os.path.join(self._container_data_dir, str_path))
            if self._spec.array_mode == 'hdf5':
                cont = ivy.Container.from_disk_as_hdf5(full_path + '.hdf5', ivyh=ivyh)
            elif self._spec.array_mode == 'pickled':
                cont = ivy.Container.from_disk_as_pickled(full_path + '.pickled', ivyh=ivyh)
            else:
                raise Exception('array_mode must be one of [ hdf5 | pickled ],'
                                'but found {}'.format(self._spec.array_mode))
            conts.append(cont)
        return ivy.Container.concat(conts, 0, ivyh=ivyh)

    # images

    def _uint8_fn(self, filepaths_in_window, ivyh):
        imgs = list()
        # ToDo: replace this with map_fn function once implemented
        for filepath in filepaths_in_window:
            str_path = bytearray(ivyh.to_numpy(filepath).tolist()).decode()
            full_path = os.path.join(self._container_data_dir, str_path)
            img_rgba = cv2.imread(full_path, -1)
            img = ivyh.array(np.expand_dims(img_rgba.astype(np.float32), 0))/255
            imgs.append(img)
        return ivyh.concatenate(imgs, 0)

    def _float_fn(self, filepaths_in_window, ivyh):
        imgs = list()
        # ToDo: replace this with map_fn function once implemented
        for filepath in filepaths_in_window:
            str_path = bytearray(ivyh.to_numpy(filepath).tolist()).decode()
            full_path = os.path.join(self._container_data_dir, str_path)
            img_rgba = cv2.imread(full_path, -1)
            img = ivyh.array(np.frombuffer(img_rgba.tobytes(), np.float32).reshape((1,) + img_rgba.shape[:-1]))
            imgs.append(img)
        return ivyh.concatenate(imgs, 0)

    def _custom_img_fn(self, filepaths_in_window, fn, ivyh):
        imgs = list()
        for filepath in filepaths_in_window:
            str_path = bytearray(ivyh.to_numpy(filepath).tolist()).decode()
            full_path = os.path.join(self._container_data_dir, str_path)
            img_raw = cv2.imread(full_path, -1)
            img = fn(img_raw)
            imgs.append(img)
        return ivyh.concatenate(imgs, 0)

    def _str_fn(self, x, key_chain, ivyh):
        for array_str in self._spec.array_strs:
            if array_str in key_chain:
                return self._array_fn(x, ivyh)
        for float_str in self._spec.float_strs:
            if float_str in key_chain:
                return self._float_fn(x, ivyh)
        for uint8_str in self._spec.uint8_strs:
            if uint8_str in key_chain:
                return self._uint8_fn(x, ivyh)
        for i, custom_img_strs in enumerate(self._spec.custom_img_strs):
            for custom_img_str in custom_img_strs:
                if custom_img_str in key_chain:
                    return self._custom_img_fn(x, self._spec.custom_img_fns[i], ivyh)
        for i, custom_strs in enumerate(self._spec.custom_strs):
            for custom_str in custom_strs:
                if custom_str in key_chain:
                    return self._spec.custom_fns[i](x, self._container_data_dir, ivyh)
        return x

    def _load_data_from_filepath_tensors(self, container, ivyh):
        ivyh = ivy.default(ivyh, ivy)
        return container.map(lambda x, kc: self._str_fn(x, kc, ivyh))

    @staticmethod
    def _to_numpy(x, kc):
        if ivy.is_array(x):
            return ivy.to_numpy(x)
        elif isinstance(x, list):
            return [ivy.to_numpy(v) if ivy.is_array(v) else v for v in x]
        else:
            return x

    # Dataset Creation #
    # -----------------#

    def _get_map_dataset(self, starting_example, ending_example):

        # container filepaths
        container_filepaths = self._load_container_filepaths_as_lists(self._container_data_dir, starting_example,
                                                                      ending_example)
        max_seq_len = max(max([len(item) for item in container_filepaths]), self._window_size)
        if self._spec.num_sequences_to_use != -1:
            container_filepaths = container_filepaths[0:self._spec.num_sequences_to_use]

        # shuffle sequences
        if self._spec.shuffle_data:
            np.random.shuffle(container_filepaths)

        # extract sequence lengths
        if self._fixed_sequence_length:
            self._sequence_lengths = [len(container_filepaths[0])] * len(container_filepaths)
            self._windows_per_seq = self._sequence_lengths[0] - self._window_size + 1
            # windowing values
            window_idxs_per_seq = ivy.reshape(ivy.arange(self._windows_per_seq, 0, 1), (self._windows_per_seq, 1))
            gather_idxs_list = list()
            for x in window_idxs_per_seq:
                gather_idxs_list.append(ivy.expand_dims(ivy.arange(x[0] + self._window_size, x[0], 1), 0))
            gather_idxs = ivy.concatenate(gather_idxs_list, 0)
            self._gather_idxs = \
                ivy.to_numpy(ivy.reshape(gather_idxs, (self._windows_per_seq * self._window_size, 1))).tolist()
        else:
            self._sequence_lengths = [len(item) for item in container_filepaths]

        # padding to make rectangular
        container_filepaths = [item + ['']*(max_seq_len - len(item)) for item in container_filepaths]

        # maybe pre-load containers
        if self._spec.preload_containers:
            # load containers with vector data and image filepath entries
            container_slices = self._get_containers_w_filepath_img_entries_as_tensor_slices(container_filepaths)
            if self._first_frame_validity_fn is not None:
                container_slices =\
                    self._first_frame_validity_fn(container_slices, [ending_example - starting_example + 1])

            # prune unwanted chains of keys
            if 'unused_key_chains' in self._spec:
                container_slices = self._prune_unused_key_chains(container_slices)

            base_cont =\
                ivy.Container.list_stack([c[0] for c in container_slices.unstack(0, container_slices.shape[0])], 0)
            dataset = MapDataset(
                base_cont.map(self._to_numpy),
                'base',
                container_slices.shape[0],
                ivyh=ivy.get_framework('numpy'))
        else:
            # load containers with filepath entries
            base_cont = ivy.Container({'fpaths': container_filepaths})
            dataset = MapDataset(base_cont.map(self._to_numpy),
                                 'base',
                                 len(container_filepaths),
                                 ivyh=ivy.get_framework('numpy'))
            dataset = dataset.map('to_numpy',
                                  lambda cont, ivyh: cont.map(lambda x_, kc: self._to_numpy(x_, ivyh)))
            dataset = dataset.map('loaded_json',
                                  self._load_json_files,
                                  self._num_workers.loaded_json)
            dataset = dataset.map('parsed_json',
                                  self._parse_json_strings,
                                  self._num_workers.parsed_json)
            if 'unused_key_chains' in self._spec:
                dataset = dataset.map('keychain_pruned',
                                      self._prune_unused_key_chains,
                                      self._num_workers.keychain_pruned)
            if self._first_frame_validity_fn is not None:
                dataset = dataset.map('valid_first_frames',
                                      lambda x_: self._first_frame_validity_fn(x_, None),
                                      self._num_workers.valid_first_frames)
        dataset = dataset.map('windowed',
                              self._group_container_into_windowed_container,
                              self._num_workers.windowed)
        dataset = dataset.unbatch('unbatched',
                                  self._num_workers.unbatched,
                                  batch_sizes=[max(item - self._window_size + 1, 1) for item in self._sequence_lengths])
        if self._spec.shuffle_data:
            dataset = dataset.shuffle('shuffled',
                                      self._spec.shuffle_buffer_size,
                                      self._num_workers.shuffled)
        dataset = dataset.map('loaded_data',
                              self._load_data_from_filepath_tensors,
                              self._num_workers.loaded_data)
        dataset = dataset.batch('batched',
                                self._batch_size,
                                self._num_workers.batched)
        dataset = dataset.map('from_numpy',
                              lambda cont, ivyh: cont.set_framework(ivy).as_arrays(),
                              self._num_workers.from_numpy)
        if ivy.exists(self._spec.post_proc_fn):
            dataset = dataset.map('post_processed',
                                  self._spec.post_proc_fn,
                                  self._num_workers.post_processed,
                                  ivyh=ivy)
        return dataset

    # Public Methods #
    # ---------------#

    def get_next_batch(self, dataset_key):
        # ToDo: explore compiling the load data method
        if dataset_key == 'training':
            return next(self._training_iterator)
        elif dataset_key == 'validation':
            return next(self._validation_iterator)
        else:
            raise Exception('invalid key')

    def get_next_training_batch(self):
        return self.get_next_batch('training')

    def get_next_validation_batch(self):
        return self.get_next_batch('validation')

    def get_dummy_batch(self):
        if self._dummy_batch is None:
            self._dummy_batch = self.get_next_training_batch()
        return self._dummy_batch.to_random()

    def close(self):
        self._training_iterator.close()
        if ivy.exists(self._validation_iterator):
            self._validation_iterator.close()

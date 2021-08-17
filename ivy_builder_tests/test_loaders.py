# global
import os
import time

import ivy
from ivy_tests import helpers

import pytest
import numpy as np

# local
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.data_loaders.json_data_loader import JSONDataLoader
from ivy_builder.data_loaders.specs.json_data_loader_spec import JSONDataLoaderSpec


@pytest.mark.parametrize(
    "preload_containers", [True, False])
@pytest.mark.parametrize(
    "array_mode", ['hdf5', 'pickled'])
@pytest.mark.parametrize(
    "prefetch_method", ['thread', 'process'])
def test_json_loader_fixed_seq_len(dev_str, f, call, preload_containers, array_mode, prefetch_method):

    if call not in [helpers.torch_call, helpers.tf_call]:
        pytest.skip()

    print('\n\nTEST {} {} {} {}'.format(ivy.current_framework_str(), preload_containers, array_mode, prefetch_method))
    test_start_time = time.perf_counter()

    # seed
    f.seed(0)
    np.random.seed(0)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=2)
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=1, window_size=1, num_sequences_to_use=1,
                                          num_training_sequences=1, preload_containers=preload_containers,
                                          array_mode=array_mode, array_strs=['array'], float_strs=['depth'],
                                          uint8_strs=['rgb'], prefetch_method=prefetch_method, shuffle_data=False)

    # data loader
    a = time.perf_counter()
    data_loader = JSONDataLoader(data_loader_spec)
    b = time.perf_counter()
    print('initializing loader: {}'.format(b - a))

    # testing
    for i in range(2):
        a = time.perf_counter()
        train_batch = data_loader.get_next_batch('training')
        b = time.perf_counter()
        print('getting batch: {}'.format(b - a))
        assert train_batch.actions.shape == (1, 1, 6)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert train_batch.array.data.shape == (1, 1, 3)

    # close
    a = time.perf_counter()
    data_loader.close()
    b = time.perf_counter()
    print('deleting data loader: {}'.format(b - a))
    print('total time: {}'.format(time.perf_counter() - test_start_time))


@pytest.mark.parametrize(
    "preload_containers", [True, False])
@pytest.mark.parametrize(
    "array_mode", ['hdf5', 'pickled'])
@pytest.mark.parametrize(
    "prefetch_method", ['thread', 'process'])
def test_json_loader(dev_str, f, call, preload_containers, array_mode, prefetch_method):

    # seed
    f.seed(0)
    np.random.seed(0)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=[2, 3, 2, 3, 3, 2])
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=3, window_size=2, num_sequences_to_use=6,
                                          num_training_sequences=3, preload_containers=preload_containers,
                                          array_mode=array_mode, array_strs=['array'], float_strs=['depth'],
                                          uint8_strs=['rgb'], num_to_prefetch=0, prefetch_method=prefetch_method,
                                          shuffle_data=False)

    # data loader
    data_loader = JSONDataLoader(data_loader_spec)

    # testing
    for i in range(2):
        train_batch = data_loader.get_next_batch('training')
        assert train_batch.actions.shape == (3, 2, 6)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert train_batch.array.data.shape == (3, 2, 3)
        valid_batch = data_loader.get_next_batch('validation')
        assert valid_batch.actions.shape == (3, 2, 6)
        assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert valid_batch.array.data.shape == (3, 2, 3)

    # delete
    data_loader.close()
    del data_loader

    # test keychain pruning, no container pre-loading, and padded windowing
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=3, window_size=3, num_sequences_to_use=6,
                                          num_training_sequences=3, preload_containers=preload_containers,
                                          array_mode=array_mode, shuffle_buffer_size=3, num_to_prefetch=0,
                                          unused_key_chains=['observations/image/ego/ego_cam_px/depth'],
                                          array_strs=['array'], float_strs=['depth'], uint8_strs=['rgb'],
                                          prefetch_method=prefetch_method, shuffle_data=False)
    data_loader = JSONDataLoader(data_loader_spec)

    train_batch = data_loader.get_next_batch('training')
    assert train_batch.actions.shape == (3, 3, 6)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert 'depth' not in train_batch.observations.image.ego.ego_cam_px
    assert train_batch.array.data.shape == (3, 3, 3)
    valid_batch = data_loader.get_next_batch('validation')
    assert valid_batch.actions.shape == (3, 3, 6)
    assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert valid_batch.array.data.shape == (3, 3, 3)

    # delete
    data_loader.close()
    del data_loader

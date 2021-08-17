# global
import ivy
import time
import math
import queue
import numbers
import threading
import numpy as np
import torch.multiprocessing as multiprocessing


# noinspection PyMissingConstructor
class Cache:
    
    def __init__(self, max_size, list_in, dict_in, lock):
        self._max_size = max_size
        self._used_keys = list_in
        self._dict = dict_in
        self._lock = lock

    def __setitem__(self, key, value):
        self._lock.acquire()
        if key in self._dict:
            self._used_keys.remove(key)
            self._used_keys.append(key)
            self._lock.release()
            return
        self._used_keys.append(key)
        if len(self._used_keys) > self._max_size:
            key_to_del = self._used_keys.pop(0)
            del self._dict[key_to_del]
        self._dict[key] = value.to_dict()
        self._lock.release()
        
    def __getitem__(self, item):
        self._lock.acquire()
        try:
            ret = self._dict[item]
        except KeyError as e:
            self._lock.release()
            raise e
        self._lock.release()
        return ivy.Container(ret)

    def __contains__(self, key):
        self._lock.acquire()
        ret = key in self._dict
        self._lock.release()
        return ret

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        self._lock.acquire()
        ret = dict(self._dict)
        self._lock.release()
        return ret

    # Getters and Setters #
    # --------------------#

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, max_size):
        self._max_size = max_size


class IteratorDataset:

    def __init__(self, base_dataset, name, size, with_prefetching=True, prefetch_timeout=None,
                 parallel_method='thread', to_gpu=None, ivyh=None):

        # framework
        self._ivy = ivy.default(ivyh, ivy)

        # gpu
        self._to_gpu = False if to_gpu in [None, False] else to_gpu
        if self._to_gpu:
            if self._to_gpu is True:
                self._to_gpu = 'cuda:0'
            elif isinstance(self._to_gpu, int):
                self._to_gpu = 'cuda:{}'.format(to_gpu)
            elif isinstance(self._to_gpu, str):
                self._to_gpu = to_gpu
            else:
                raise Exception('to_gpu must be an int, str, None, True, or False, but found {}'.format(to_gpu))

        # config
        self._name = name
        self._size = size

        # base dataset
        self._base_dataset = base_dataset

        # base dataset iterator
        self._base_dataset_iterator = iter(base_dataset)

        # pre-fetch sub-process
        self._with_prefetching = with_prefetching
        self._prefetch_timeout = prefetch_timeout
        self._parallel_method = parallel_method
        self._prefetch_running = False
        if self._with_prefetching:
            if self._parallel_method == 'process':
                self._input_queue = multiprocessing.Queue()
                self._output_queue = multiprocessing.Queue()
                self._worker = multiprocessing.Process(
                    target=self._process_worker_fn, args=(base_dataset, self._base_dataset_iterator, self._input_queue,
                                                          self._output_queue))
                self._get_next = self._get_from_process
            elif self._parallel_method == 'thread':
                self._thread = threading.Thread(target=self._thread_worker_fn)
                self._lock_for_next = threading.Lock()
                self._lock_for_spin = threading.Lock()
                self._keep_spinning = True
                self._next = None
                self._get_next = self._get_from_thread
            else:
                raise Exception('parallel method must be one of [ process | thread ], but found {}'.format(
                    self._parallel_method))

    # Private #
    # --------#

    @staticmethod
    def _process_worker_fn(base_dataset, base_dataset_iterator, input_queue, output_queue):
        keep_going = True
        while keep_going:
            try:
                keep_going = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            output_queue.put(next(base_dataset_iterator).to_dict())
        # with open("{}_{}.log".format(base_dataset.name, id(base_dataset)), "a+") as f:
        #     f.write("closing dataset: {}\n".format(time.perf_counter()))
        base_dataset.close()
        # with open("{}_{}.log".format(base_dataset.name, id(base_dataset)), "a+") as f:
        #     f.write("dataset closed: {}\n".format(time.perf_counter()))
        return

    def _thread_worker_fn(self):
        while True:
            time.sleep(0.01)
            self._lock_for_next.acquire()
            if not ivy.exists(self._next):
                next_data = next(self._base_dataset_iterator)
                self._next = next_data.to_dev(self._to_gpu) if self._to_gpu else next_data
            self._lock_for_next.release()
            self._lock_for_spin.acquire()
            if not self._keep_spinning:
                self._lock_for_spin.release()
                break
            self._lock_for_spin.release()

    def _get_from_thread(self):
        time_taken = 0
        while True:
            self._lock_for_next.acquire()
            if ivy.exists(self._next):
                self._lock_for_next.release()
                break
            self._lock_for_next.release()
            time.sleep(0.01)
            time_taken += 0.01
            if ivy.exists(self._prefetch_timeout) and time_taken > self._prefetch_timeout:
                raise Exception('Prefetch timed out')
        self._lock_for_next.acquire()
        ret = self._next
        self._next = None
        self._lock_for_next.release()
        return ret

    def _get_from_process(self):
        self._input_queue.put(True)
        next_data = ivy.Container(self._output_queue.get(timeout=self._prefetch_timeout), ivyh=self._ivy)
        if self._to_gpu:
            next_data = next_data.to_dev(self._to_gpu)
        return next_data

    def _start_prefetching(self):
        if self._parallel_method == 'process':
            self._worker.start()
            self._input_queue.put(True)
        else:
            self._thread.start()

    def __next__(self):
        if not self._with_prefetching:
            next_data = next(self._base_dataset_iterator)
            if self._to_gpu:
                next_data = next_data.to_dev(self._to_gpu)
            return next_data
        if not self._prefetch_running:
            self._start_prefetching()
            self._prefetch_running = True
        return self._get_next()

    def __del__(self):
        self.close()

    def close(self):
        if not isinstance(self._base_dataset, ivy.Container):
            self._base_dataset.close()
        if self._prefetch_running:
            if self._parallel_method == 'process':
                try:
                    self._input_queue.put(False)
                    if self._worker.is_alive():
                        # ToDo: increase this timeout once the blocking issue for json data loader is fixed
                        self._worker.join(timeout=0.1)
                    self._input_queue.cancel_join_thread()
                    self._input_queue.close()
                    self._output_queue.cancel_join_thread()
                    self._output_queue.close()
                finally:
                    if self._worker.is_alive():
                        self._worker.terminate()
            else:
                self._lock_for_spin.acquire()
                self._keep_spinning = False
                self._lock_for_spin.release()
                if self._thread.is_alive():
                    self._thread.join()
        self._prefetch_running = False


class MapDataset:

    def __init__(self, base_dataset, name, size, base_slice_fn=None, trans_fn=None, slice_fn=None,
                 elementwise_query_fn=True, cache_size=0, cache=None, num_processes=1,
                 queue_timeout=None, blocking_retreival=True, is_subprocess=False, ivyh=None):
        self._name = name
        self._size = size
        self._base_slice_fn = base_slice_fn
        if base_slice_fn is None:
            self._slice_base_dataset = self._default_base_slice_fn
        else:
            self._slice_base_dataset = base_slice_fn
        self._trans_fn = trans_fn
        self._slice_fn = slice_fn
        if slice_fn is None:
            self._slice_dataset = self._default_slice_fn
        else:
            self._slice_dataset = slice_fn
        self._elementwise_query_fn = elementwise_query_fn
        self._with_caching = cache_size > 0
        self._cache_size = cache_size
        if cache:
            self._cache = cache
        else:
            manager = multiprocessing.Manager()
            shared_list = manager.list()
            shared_dict = manager.dict()
            lock = multiprocessing.Lock()
            self._cache = Cache(self._cache_size, shared_list, shared_dict, lock)

        self._num_processes = multiprocessing.cpu_count() if num_processes is None else num_processes
        self._blocking_retreival = blocking_retreival
        self._queue_timeout = queue_timeout
        self._is_subprocess = is_subprocess
        self._ivy = ivy.default(ivyh, ivy)
        if isinstance(base_dataset, ivy.Container):
            base_dataset.set_framework(ivyh)
        self._base_dataset = base_dataset
        self._workers_initialized = False
        self._has_workers = False

    # Private #
    # --------#

    def _deep_copy(self, num_processes=None, shared_list=None, shared_dict=None):
        # noinspection PyProtectedMember
        return MapDataset(
            base_dataset=self._base_dataset.copy() if isinstance(self._base_dataset, ivy.Container)
            else self._base_dataset._deep_copy(), name=self._name, size=self._size,
            base_slice_fn=self._base_slice_fn, trans_fn=self._trans_fn, slice_fn=self._slice_fn,
            elementwise_query_fn=self._elementwise_query_fn, cache_size=self._cache_size, cache=self._cache,
            num_processes=ivy.default(num_processes, self._num_processes), queue_timeout=self._queue_timeout,
            blocking_retreival=self._blocking_retreival, is_subprocess=True)

    def _initialize_all_workers(self):
        if not self._workers_initialized and self._num_processes > 1:
            self._workers = list()
            self._slice_queues = list()
            self._output_queues = list()
            for i in range(self._num_processes):
                dataset_copy = self._deep_copy(1)
                index_queue = multiprocessing.Queue()
                output_queue = multiprocessing.Queue()
                worker = multiprocessing.Process(
                    target=self._worker_fn, args=(index_queue, output_queue, dataset_copy))
                worker.start()
                self._slice_queues.append(index_queue)
                self._output_queues.append(output_queue)
                self._workers.append(worker)
            self._has_workers = True
        self._workers_initialized = True

    @staticmethod
    def _worker_fn(index_queue, output_queue, dataset):
        # with open("{}_{}.log".format(dataset.name, id(dataset)), "a+") as f:
        #     f.write("worker started: {}\n".format(time.perf_counter()))
        while True:
            try:
                slice_obj = index_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if slice_obj is None:
                # ToDo: work out why this command below works, but del dataset hangs, despite only calling
                #  close(), perhaps processes have trouble explicitly deleting arguments passed in?
                # with open("{}_{}.log".format(dataset.name, id(dataset)), "a+") as f:
                #     f.write("closing dataset: {}\n".format(time.perf_counter()))
                dataset.close()
                # with open("{}_{}.log".format(dataset.name, id(dataset)), "a+") as f:
                #     f.write("dataset closed, worker exited: {}\n".format(time.perf_counter()))
                return
            item = dataset[slice_obj]
            output_queue.put(item.to_dict())

    @staticmethod
    def _empty_queue(queue_in):
        while True:
            try:
                queue_in.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _is_int(val):
        return abs(round(val) - val) < 1e-6

    @staticmethod
    def _ensure_number_is_int(val):
        val_rounded = round(val)
        if abs(val_rounded - val) > 1e-6:
            raise Exception('Trying to slice ivy Container with non-integer slice {}'.format(val))
        return int(val_rounded)

    @staticmethod
    def _slice_dataset(slice_obj, dataset):
        if isinstance(dataset, ivy.Container):
            if isinstance(slice_obj, numbers.Number):
                slice_obj = MapDataset._ensure_number_is_int(slice_obj)
            else:
                so_start = MapDataset._ensure_number_is_int(slice_obj.start)
                so_stop = MapDataset._ensure_number_is_int(slice_obj.stop)
                if slice_obj.step is None:
                    so_step = 1
                else:
                    so_step = MapDataset._ensure_number_is_int(slice_obj.step)
                slice_obj = slice(so_start, so_stop, so_step)
            return dataset[slice_obj]
        else:
            return dataset[slice_obj]

    @staticmethod
    def _default_base_slice_fn(slice_obj, dataset):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = slice(slice_obj, slice_obj+1, 1)
        ret = MapDataset._slice_dataset(slice_obj, dataset)
        return ret

    @staticmethod
    def _default_slice_fn(slice_obj, sliced_dataset, dataset_size):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = 0
        else:
            if slice_obj.stop > slice_obj.start:
                slice_size = slice_obj.stop - slice_obj.start
            else:
                slice_size = slice_obj.stop + dataset_size - slice_obj.start
            slice_obj = slice(0, slice_size, slice_obj.step)
        return MapDataset._slice_dataset(slice_obj, sliced_dataset)

    def _get_base_item(self, slice_obj):
        base_dataset = self._slice_base_dataset(slice_obj, self._base_dataset)
        if self._trans_fn is not None:
            if self._elementwise_query_fn:
                vals = [self._trans_fn(base_dataset[i], self._ivy) for i in range(base_dataset.shape[0])]
                return ivy.Container.list_stack(vals, 0)
            return self._trans_fn(base_dataset, self._ivy)
        return base_dataset

    def _get_item_from_slice_objs(self, base_slice_objs, slice_obj):
        if len(base_slice_objs) == 1:
            item = self._get_base_item(base_slice_objs[0])
        else:
            item = ivy.Container.list_join([self._get_base_item(bso) for bso in base_slice_objs])
        return self._slice_dataset(slice_obj, item, self._size)

    def _wrap_slice_obj(self, slice_obj):
        if isinstance(slice_obj, numbers.Number):
            return slice_obj % self._size
        else:
            so_start_orig = slice_obj.start
            so_stop_orig = slice_obj.stop
            so_start_wrapped = so_start_orig % self._size
            if abs(so_stop_orig - so_start_orig - 1) < 1e-6:
                return slice(so_start_wrapped, so_start_wrapped + 1, 1)
            so_stop_wrapped = so_stop_orig % self._size
            if abs(so_stop_wrapped) < 1:
                so_stop_wrapped = self._size + so_stop_wrapped
            return slice(so_start_wrapped, so_stop_wrapped, 1)

    def _wrap_base_slice_obj(self, slice_obj):
        if isinstance(slice_obj, numbers.Number):
            return [slice_obj]
        elif slice_obj.stop < slice_obj.start:
            end_idx_0 = slice_obj.start + math.ceil(self._size - slice_obj.start)
            slice_obj_0 = slice(slice_obj.start, end_idx_0, 1)
            start_idx_1 = end_idx_0 - self._size
            slice_obj_1 = slice(start_idx_1, slice_obj.stop, 1)
            return [slice_obj_0, slice_obj_1]
        return [slice_obj]

    def _add_to_cache(self, so, item):
        if isinstance(so, numbers.Number):
            self._cache[so] = item
        else:
            for i in np.arange(so.start, so.stop-1e-3, 1.):
                self._cache[i] = MapDataset._slice_dataset(i - so.start, item)

    def __del__(self):
        self.close()

    def _get_item_after_cache_n_wrap(self, slice_obj):
        base_slice_obj = self._wrap_base_slice_obj(slice_obj)
        ret = self._get_item_from_slice_objs(base_slice_obj, slice_obj)
        return ret

    def _get_item(self, slice_obj):
        # ToDo: simplify this method for the case of no caching
        slice_obj = self._wrap_slice_obj(slice_obj)
        slice_objs = self._wrap_base_slice_obj(slice_obj)
        items = list()
        range_to_iterate = [i for s in [np.arange(so, so+1, 1) if isinstance(so, numbers.Number) else
                            np.arange(so.start, so.stop, 1) for so in slice_objs] for i in s]
        start = range_to_iterate[0]
        pre_cache_items_exist = False
        for i, so in enumerate(range_to_iterate):
            so = int(so) if self._is_int(so) else so
            cache_item = None
            try:
                cache_item = self._cache[so].map(lambda x, kc: [x])
            except KeyError:
                pre_cache_items_exist = True
            from_cache = ivy.exists(cache_item)
            if from_cache:
                if pre_cache_items_exist:
                    slc = slice(start, self._size if so == 0 else so, 1)
                    item = self._get_item_after_cache_n_wrap(slc)
                    if self._with_caching:
                        self._add_to_cache(slc, item)
                    items.append(item)
                    pre_cache_items_exist = False
                    items.append(cache_item)
                else:
                    items.append(cache_item)
                start = (so + 1) % self._size
            elif i == len(range_to_iterate) - 1 and not from_cache:
                slc = slice(start, so+1, 1)
                item = self._get_item_after_cache_n_wrap(slc)
                if self._with_caching:
                    self._add_to_cache(slc, item)
                items.append(item)
        if len(items) == 1:
            # ToDo: determine whether the map should be applied to both below
            if isinstance(slice_obj, numbers.Number):
                return items[0][0]
            return items[0].map(lambda x, kc: x if isinstance(x, list) else [x])
        items_as_lists = [item.map(lambda x, kc: x if isinstance(x, list) else [x]) for item in items]
        return ivy.Container.list_join(items_as_lists)

    # Public #
    # -------#

    def __getitem__(self, slice_obj):
        if self._num_processes < 2 or isinstance(slice_obj, numbers.Number):
            ret = self._get_item(slice_obj)
            return ret
        if not self._workers_initialized:
            self._initialize_all_workers()
        slice_size = int(round(slice_obj.stop - slice_obj.start))
        num_sub_slices = min(slice_size, self._num_processes)
        slice_points = np.linspace(slice_obj.start, slice_obj.stop, num_sub_slices+1)
        slice_sizes = (slice_points[1:] - slice_points[:-1]).astype(np.int32)
        if MapDataset._is_int(slice_obj.start) and MapDataset._is_int(slice_obj.stop):
            slice_points = np.round(slice_points)
        sub_slices = [slice(slice_points[i], slice_points[i+1], 1.) for i in range(num_sub_slices)]
        # ToDo: generalize the line below to non-integer slices
        offset = slice_obj.start % self._num_processes
        # ToDo: improve the stability, emptying these queues does not guarentee a process will write to it which
        #  corresponds to a previous set of slices
        [self._empty_queue(q) for q in self._output_queues]
        [self._slice_queues[int((i + offset) % self._num_processes)].put(sub_slice)
         for i, sub_slice in enumerate(sub_slices)]
        output_queues = [self._output_queues[int((i + offset) % self._num_processes)] for i in range(num_sub_slices)]
        if self._blocking_retreival:
            items_as_lists = [ivy.Container(q.get(timeout=self._queue_timeout), ivyh=self._ivy) for q in output_queues]
            return ivy.Container.list_join(items_as_lists)
        cont = ivy.Container(queues=output_queues, queue_load_sizes=slice_sizes)
        return cont

    def map(self, name, map_func, num_processes=1, queue_timeout=None, base_slice_fn=None, ivyh=None):
        return MapDataset(base_dataset=self,
                          name=name,
                          size=self._size,
                          base_slice_fn=base_slice_fn,
                          trans_fn=map_func,
                          cache_size=self._cache_size,
                          num_processes=num_processes,
                          queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                          ivyh=ivy.default(ivyh, self._ivy))

    def batch(self, name, batch_size, num_processes=1, queue_timeout=None, ivyh=None):
        def batch_array(x, ivyh_):
            return [ivyh_.concatenate(
                [ivyh_.expand_dims(item, 0) for item in x[i*batch_size:i*batch_size+batch_size]], 0)
                for i in range(int(len(x)/batch_size))]

        def batch_cont(cont, ivyh_):
            return cont.map(lambda x, kc: batch_array(x, ivyh_))

        def base_slice_fn(slc_obj, dataset):
            if isinstance(slc_obj, numbers.Number):
                base_slice_obj =\
                    slice(int(round(batch_size * slc_obj)), int(round(batch_size * slc_obj + batch_size)), 1)
            else:
                so_start = int(round(batch_size * slc_obj.start))
                so_stop = int(round(batch_size * slc_obj.stop))
                base_slice_obj = slice(so_start, so_stop, 1)
            return MapDataset._slice_dataset(base_slice_obj, dataset)

        return MapDataset(base_dataset=self,
                          name=name,
                          size=float(self._size / batch_size),
                          base_slice_fn=base_slice_fn,
                          trans_fn=batch_cont,
                          elementwise_query_fn=False,
                          cache_size=int(math.ceil(self._cache_size / batch_size)),
                          num_processes=num_processes,
                          queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                          ivyh=ivy.default(ivyh, self._ivy))

    def unbatch(self, name, num_processes=1, queue_timeout=None, ivyh=None, cache_size=None, batch_sizes=None):

        unbatch_slice_dict = dict()
        slice_dict = dict()
        size_so_far = 0
        size = math.ceil(self._size)
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]*size
        for i in range(size):
            if batch_sizes is None:
                data = self._get_item(i)
                data_size = data.shape[0]
            else:
                data_size = batch_sizes[i]
            if i == size - 1 and self._size % 1 != 0:
                data_size = int(round(data_size * (self._size - math.floor(self._size))))
            for j in range(data_size):
                unbatch_slice_dict[size_so_far + j] = i
                slice_dict[size_so_far + j] = j
            size_so_far += data_size
        unrolled_size = size_so_far

        def base_slice_fn(slice_obj, dataset):
            if isinstance(slice_obj, numbers.Number):
                slice_obj = slice(slice_obj, slice_obj + 1, 1)
            so_start = unbatch_slice_dict[slice_obj.start]
            so_stop = unbatch_slice_dict[slice_obj.stop - 1] + 1
            so_stop = so_stop + 1 if so_stop == so_start else so_stop
            so = slice(so_start, so_stop, 1)
            return MapDataset._slice_dataset(so, dataset)

        def unbatch_fn(cont, ivyh_):
            return cont.map(lambda x, kc: [c for o in [ivyh_.unstack(item, 0) for item in x] for c in o])

        def slice_fn(slice_obj, sliced_dataset, dataset_size):
            if isinstance(slice_obj, numbers.Number):
                return MapDataset._slice_dataset(slice_dict[slice_obj], sliced_dataset)
            else:
                if slice_obj.stop > slice_obj.start:
                    slice_size = slice_obj.stop - slice_obj.start
                else:
                    slice_size = slice_obj.stop + unrolled_size - slice_obj.start
                so_start = slice_dict[slice_obj.start]
                so_stop = so_start + slice_size
                so = slice(so_start, so_stop, 1)
                return MapDataset._slice_dataset(so, sliced_dataset)

        return MapDataset(base_dataset=self,
                          name=name,
                          size=unrolled_size,
                          base_slice_fn=base_slice_fn,
                          trans_fn=unbatch_fn,
                          slice_fn=slice_fn,
                          elementwise_query_fn=False,
                          cache_size=int(math.ceil(self._cache_size * unrolled_size / self._size))
                                if cache_size is None else cache_size,
                          num_processes=num_processes,
                          queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                          ivyh=ivy.default(ivyh, self._ivy))

    def shuffle(self, name, shuffle_buffer_size, num_processes=1, queue_timeout=None, ivyh=None):
        if shuffle_buffer_size == 0:
            return self
        pre_shuffled = self.batch('pre_' + name,
                                  shuffle_buffer_size,
                                  num_processes=num_processes,
                                  queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                                  ivyh=ivy.default(ivyh, self._ivy))
        shuffled = MapDataset(base_dataset=pre_shuffled,
                              name=name,
                              size=pre_shuffled.size,
                              trans_fn=lambda cont, _: cont.shuffle(),
                              cache_size=self._cache_size,
                              num_processes=num_processes,
                              queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                              ivyh=ivy.default(ivyh, self._ivy))
        post_shuffled = shuffled.unbatch('post_' + name,
                                         num_processes=num_processes,
                                         queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                                         ivyh=ivy.default(ivyh, self._ivy),
                                         cache_size=self._cache_size,
                                         batch_sizes=shuffle_buffer_size)
        return post_shuffled

    def prefetch(self, name, buffer_size, ivyh=None):

        # noinspection PyUnresolvedReferences
        def base_slice_fn(slc_obj, dataset):
            if isinstance(slc_obj, numbers.Number):
                so_start = slc_obj
                so_stop = slc_obj + 1 + buffer_size
            else:
                so_start = slc_obj.start
                so_stop = slc_obj.stop + buffer_size
            base_slice_obj = slice(so_start, so_stop, 1)
            return MapDataset._slice_dataset(base_slice_obj, dataset)

        # ToDo: try to apply this to the new dataset, not the current base dataset
        self._blocking_retreival = False
        self._num_processes = buffer_size+1
        orig_cache_size = self._cache_size
        self._cache_size = max(self._cache_size, buffer_size+1)
        self._cache.max_size = self._cache_size

        return MapDataset(base_dataset=self,
                          name=name,
                          size=self._size,
                          base_slice_fn=base_slice_fn,
                          cache_size=orig_cache_size,
                          num_processes=1,
                          ivyh=ivy.default(ivyh, self._ivy))

    def to_gpu(self, name, num_processes=1, queue_timeout=None, gpu_idx=0):

        def item_to_gpu(x, ivyh_):
            return ivyh_.array(x, dev_str='gpu:' + str(gpu_idx))

        def cont_to_gpu(cont, ivyh_):
            return cont.map(lambda x, kc: item_to_gpu(x, ivyh_))

        return MapDataset(base_dataset=self,
                          name=name,
                          size=self._size,
                          trans_fn=cont_to_gpu,
                          cache_size=self._cache_size,
                          num_processes=num_processes,
                          queue_timeout=ivy.default(queue_timeout, self._queue_timeout),
                          ivyh=self._ivy)

    def to_iterator(self, name, with_prefetching=True, prefetch_timeout=None, parallel_method='thread', to_gpu=False,
                    ivyh=None):
        return IteratorDataset(base_dataset=self,
                               name=name,
                               size=self._size,
                               with_prefetching=with_prefetching,
                               prefetch_timeout=ivy.default(prefetch_timeout, self._queue_timeout),
                               parallel_method=parallel_method,
                               to_gpu=to_gpu,
                               ivyh=ivy.default(ivyh, self._ivy))

    def close(self):
        if not isinstance(self._base_dataset, ivy.Container):
            self._base_dataset.close()
        if self._has_workers:
            try:
                for i, w in enumerate(self._workers):
                    self._slice_queues[i].put(None)
                    if w.is_alive():
                        w.join(timeout=1.0)
                for q in self._slice_queues:
                    q.cancel_join_thread()
                    q.close()
                for q in self._output_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                for w in self._workers:
                    if w.is_alive():
                        w.terminate()
        # This line below is only needed because close() is called explicitly from inside the worker_fn.
        #  If the dataset can be deleted directly from inside worker_fn, then this subsequent delete will not be called.
        self._has_workers = False

    # Getters #
    # --------#

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

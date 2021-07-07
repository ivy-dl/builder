# global
import os
import ivy
import json
import logging
import numpy as np
from ray import tune
import multiprocessing
from ray.tune import CLIReporter
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler

# local
from ivy.core.container import Container
from ivy_builder import builder as builder_module
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs.data_loader_spec import DataLoaderSpec
from ivy_builder.specs.network_spec import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.specs.tuner_spec import TunerSpec

SPEC_KEYS = ['dd', 'ds', 'dls', 'ns', 'ts']


def _convert_tuner_spec(config):
    return_dict = dict()

    for i, (key, arg) in enumerate(config.items()):
        spec_key = [spec_key for spec_key in SPEC_KEYS if spec_key in key]
        if not spec_key:
            return_dict[key] = arg
            continue
        min_val = arg['min']
        max_val = arg['max']
        mean_val = (max_val + min_val) / 2
        sd_val = max_val - mean_val
        gaussian = 'gaussian' in arg and arg['gaussian']
        uniform = 'uniform' in arg and arg['uniform']
        grid = 'grid' in arg and arg['grid']
        exponential = 'exponential' in arg and arg['exponential']
        as_int = 'as_int' in arg and arg['as_int']
        if gaussian:
            if exponential:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        int(np.round(np.exp(np.random.normal(np.log(m), np.log(s)))))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        np.exp(np.random.normal(np.log(m), np.log(s)))
            else:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        int(np.round(np.random.normal(m, s)))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        np.random.normal(m, s)
            return_dict[key] = tune.sample_from(sample_func)
        elif uniform:
            if exponential:
                if as_int:
                    sample_func = lambda spec, mi=min_val, ma=max_val:\
                        int(np.round(np.exp(np.random.uniform(np.log(mi), np.log(ma)))))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val:\
                        np.exp(np.random.uniform(np.log(mi), np.log(ma)))
            else:
                if as_int:
                    sample_func = lambda spec, mi=min_val, ma=max_val: int(np.round(np.random.uniform(mi, ma)))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val: np.random.uniform(mi, ma)
            return_dict[key] = tune.sample_from(sample_func)
        elif grid:
            num_samples = arg['num_samples']
            if exponential:
                if as_int:
                    grid_vals = np.round(np.exp(np.linspace(
                        np.log(min_val), np.log(max_val), num_samples))).astype(np.int32).tolist()
                else:
                    grid_vals = np.round(np.exp(np.linspace(np.log(min_val), np.log(max_val), num_samples))).tolist()
            else:
                if as_int:
                    grid_vals = np.round(np.linspace(min_val, max_val, num_samples)).astype(np.uint32).tolist()
                else:
                    grid_vals = np.linspace(min_val, max_val, num_samples).tolist()
            return_dict[key] = tune.grid_search(grid_vals)
        else:
            raise Exception('invalid mode, one of [ gaussian | uniform | grid ] must be selected.')
    return Container(return_dict)


class Tuner:

    def __init__(self,
                 data_loader_class,
                 network_class,
                 trainer_class,
                 dataset_dirs_args: dict = None,
                 dataset_dirs_class: DatasetDirs.__base__ = DatasetDirs,
                 dataset_spec_args: dict = None,
                 dataset_spec_class: DatasetSpec.__base__ = DatasetSpec,
                 data_loader_spec_args: dict = None,
                 data_loader_spec_class: DataLoaderSpec.__base__ = DataLoaderSpec,
                 network_spec_args: dict = None,
                 network_spec_class: NetworkSpec.__base__ = NetworkSpec,
                 trainer_spec_args: dict = None,
                 trainer_spec_class: TrainerSpec.__base__ = TrainerSpec,
                 tuner_spec_args: dict = None,
                 tuner_spec_class: TrainerSpec.__base__ = TunerSpec):
        """
        base class for any tune trainers
        """
        self._data_loader_class = data_loader_class
        self._network_class = network_class
        self._trainer_class = trainer_class
        self._dataset_dirs_args = dataset_dirs_args
        self._dataset_dirs_class = dataset_dirs_class
        self._dataset_spec_args = dataset_spec_args
        self._dataset_spec_class = dataset_spec_class
        self._data_loader_spec_args = data_loader_spec_args
        self._data_loader_spec_class = data_loader_spec_class
        self._network_spec_args = network_spec_args
        self._network_spec_class = network_spec_class
        self._trainer_spec_args = trainer_spec_args
        self._trainer_spec_class = trainer_spec_class
        self._tuner_spec_args = tuner_spec_args
        self._tuner_spec_class = tuner_spec_class

        # initialized on _setup
        self._trainer = None

        # builder
        while len(ivy.framework_stack) > 0:
            logging.info('unsetting framework {}, framework stack must be empty when'
                         'initializing tuner class.'.format(ivy.framework_stack[-1]))
            ivy.unset_framework()
        self._builder = builder_module

    def tune(self, name, num_samples, parallel_trials, grace_period):

        # Create Trainable class #
        # -----------------------#

        # builder for TuneTrainable
        builder = self._builder

        # classes and args for TuneTrainable
        data_loader_class = self._data_loader_class
        network_class = self._network_class
        trainer_class = self._trainer_class
        dataset_dirs_args = self._dataset_dirs_args
        dataset_dirs_class = self._dataset_dirs_class
        dataset_spec_args = self._dataset_spec_args
        dataset_spec_class = self._dataset_spec_class
        data_loader_spec_args = self._data_loader_spec_args
        data_loader_spec_class = self._data_loader_spec_class
        network_spec_args = self._network_spec_args
        network_spec_class = self._network_spec_class
        trainer_spec_args = self._trainer_spec_args
        trainer_spec_args['save_freq'] = -1
        trainer_spec_args['with_writer'] = False
        trainer_spec_class = self._trainer_spec_class
        tuner_spec_args = self._tuner_spec_args
        tuner_spec_class = self._tuner_spec_class

        class TuneTrainable(tune.Trainable):

            def setup(self, _):
                ivy.set_framework(self.config['framework'])
                self.timestep = 0
                self._trainer_global_step = 0
                self._train_steps_per_tune_step = self.config['train_steps_per_tune_step']

                new_args = dict()
                for class_key, args in zip(SPEC_KEYS, [dataset_dirs_args, dataset_spec_args, data_loader_spec_args,
                                                       network_spec_args, trainer_spec_args]):
                    if args:
                        new_args[class_key] = dict([(key, self.config[class_key + '_' + key])
                                                    if class_key + '_' + key in self.config
                                                    else (key, value) for key, value in args.items()])
                    else:
                        new_args[class_key] = None

                self._trainer = builder.build_trainer(data_loader_class,
                                                      network_class,
                                                      trainer_class,
                                                      new_args['dd'],
                                                      dataset_dirs_class,
                                                      new_args['ds'],
                                                      dataset_spec_class,
                                                      new_args['dls'],
                                                      data_loader_spec_class,
                                                      new_args['ns'],
                                                      network_spec_class,
                                                      new_args['ts'],
                                                      trainer_spec_class)
                self._trainer.setup()

            def step(self):
                self._trainer_global_step = self._trainer.train(self._trainer_global_step,
                                                                self._trainer_global_step +
                                                                self._train_steps_per_tune_step)
                self.timestep += 1
                return {'cost': ivy.to_numpy(self._trainer.moving_average_loss)}

            def save_checkpoint(self, checkpoint_dir):
                path = os.path.join(checkpoint_dir, 'checkpoint')
                with open(path, "w") as f:
                    f.write(json.dumps({"timestep": self.timestep}))
                self._trainer.save(checkpoint_dir)
                return path

            def load_checkpoint(self, checkpoint_path):
                with open(checkpoint_path) as f:
                    self.timestep = json.loads(f.read())["timestep"]
                self._trainer.restore(checkpoint_path, self._trainer_global_step)

            def cleanup(self):
                ivy.unset_framework()

        # Build local tune specification #
        # -------------------------------#

        # tuner spec
        ivy.set_framework(self._tuner_spec_args['framework'])
        tuner_spec = self._builder.build_tuner_spec(
            self._data_loader_class, self._network_class, self._trainer_class, self._dataset_dirs_args,
            self._dataset_dirs_class, self._dataset_spec_args, self._dataset_spec_class, self._data_loader_spec_args,
            self._data_loader_spec_class, self._network_spec_args, self._network_spec_class, self._trainer_spec_args,
            self._trainer_spec_class, self._tuner_spec_args, self._tuner_spec_class)
        tuner_spec = _convert_tuner_spec(tuner_spec)

        # Run this trainable class #
        # -------------------------#

        ahb = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="cost",
            mode="min",
            grace_period=grace_period,
            max_t=int(np.ceil(tuner_spec.trainer.spec.total_iterations/tuner_spec.train_steps_per_tune_step)))

        num_cpus = multiprocessing.cpu_count()
        num_gpus = ivy.num_gpus()
        cpus_per_trial = num_cpus/parallel_trials
        gpus_per_trial = num_gpus/parallel_trials
        ivy.unset_framework()

        reporter = CLIReporter(['cost'])

        tune.run(TuneTrainable,
                 progress_reporter=reporter,
                 name=name,
                 scheduler=ahb,
                 stop={"training_iteration":
                           int(np.ceil(tuner_spec.trainer.spec.total_iterations/tuner_spec.train_steps_per_tune_step))},
                 num_samples=num_samples,
                 resources_per_trial={
                     "cpu": cpus_per_trial,
                     "gpu": gpus_per_trial
                 },
                 config=tuner_spec,
                 local_dir=tuner_spec.trainer.spec.log_dir)

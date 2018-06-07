import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
from collections import defaultdict

class DataIterator(NumpyArrayIterator):
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png', dc_idx=0):
        super(DataIterator, self).__init__(x, y, image_data_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)
        self.dc_idx = dc_idx
        self.data_limit = len(x)




    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        self.epoch = np.int(np.ceil(self.total_batches_seen / self.image_data_generator.steps_per_epoch))
        if self.image_data_generator.curriculum:
            # sorted_indices = self.image_data_generator.sorted_indices
            # num_classes = self.image_data_generator.num_classes
            # self.data_limit = \
            self.image_data_generator.curriculum_schedule(self)#self.epoch,self.image_data_generator.history)
            self._set_index_array_cl()

            if not self.total_batches_seen % 500:
                print(self.data_limit)

            idx = idx % np.int(self.data_limit/ self.batch_size)
        else:
            if self.index_array is None:
                self._set_index_array()

        index_array = self.index_array[self.batch_size * idx:
        self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def _set_index_array_cl(self):
        self.index_array = np.arange(self.data_limit)
        np.random.shuffle(self.index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            if j in self.image_data_generator.subset_index_array:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DataGenerator(ImageDataGenerator):
    def __init__(self):
        super(DataGenerator, self).__init__()
        self.history = defaultdict(list)

    def flow(self, x, y=None, batch_size=100, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        self.data_iterator = DataIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            dc_idx=0)
        return self.data_iterator



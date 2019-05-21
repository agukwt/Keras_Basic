

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, data_paths, data_classes,
                 batch_size=1, width=256, height=256, ch=3, num_of_class=2):
        """construction

        :param data_paths: List of image file
        :param data_classes: List of class
        :param batch_size: Batch size
        :param width: Image width
        :param height: Image height
        :param ch: Num of image channels
        :param num_of_class: Num of classes
        """

        self.data_paths = data_paths
        self.data_classes = data_classes
        self.length = len(data_paths)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.ch = ch
        self.num_of_class = num_of_class
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        """Get batch data

        :param idx: Index of batch

        :return imgs: numpy array of images
        :return labels: numpy array of label
        """

        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        item_paths = self.data_paths[start_pos : end_pos]
        item_classes = self.data_classes[start_pos : end_pos]
        imgs = np.empty((len(item_paths), self.height, self.width, self.ch), dtype=np.float32)
        labels = np.empty((len(item_paths), num_of_class), dtype=np.float32)

        for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
            img, label = _load_data(item_path, item_class, self.width, self.height, self.ch)
            imgs[i, :] = img
            labels[i] = label

        return imgs, labels


    def __len__(self):
        """Batch length"""

        return self.num_batches_per_epoch


    def on_epoch_end(self):
        """Task when end of epoch"""
        pass
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size=32, seed=42):

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        self.num_classes = len(dataset.dataset.classes)

        if batch_size % self.num_classes != 0:
            raise ValueError(
                "batch_size must be divisible by number of classes"
            )

        self.samples_per_class = batch_size // self.num_classes

        self.class_indices = {}

        for index in range(len(dataset)):

            original_index = dataset.indices[index]

            label = dataset.dataset.targets[original_index]

            if label not in self.class_indices:
                self.class_indices[label] = []

            self.class_indices[label].append(index)

        self.num_batches = len(dataset) // batch_size

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        class_samples = {}

        for label in self.class_indices:

            indices = self.class_indices[label]

            selected_indices = torch.randint(
                low=0,
                high=len(indices),
                size=(self.num_batches * self.samples_per_class,),
                generator=generator
            )

            class_samples[label] = []

            for i in selected_indices:

                class_samples[label].append(
                    indices[i.item()]
                )

        for batch_number in range(self.num_batches):

            batch = []

            for label in sorted(class_samples.keys()):

                start = batch_number * self.samples_per_class
                end = start + self.samples_per_class

                batch.extend(
                    class_samples[label][start:end]
                )

            yield batch

    def __len__(self):

        return self.num_batches
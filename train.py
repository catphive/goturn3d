
import typing
import torch
import datasets

import datasets
import models
import time
import cli

class Trainer:

    dataset_list: typing.List[datasets.KittiDataset]
    model: models.LidarGoturnModel

    def __init__(self, dataset_list, model, device):

        model.to(device)

        self.dataset_list = dataset_list
        self.model = model
        self.device = device


    def train(self, num_epochs = 96, batch_size = 50, learning_rate=1e-5, momentum=0.9, weight_decay=0.0005,
              lr_decay_step=20, gamma=0.1):
        # TODO: support multiple datasets in list.
        dataset = torch.utils.data.ConcatDataset(self.dataset_list)

        print(f'dataset size = {len(dataset)}')

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        criterion = torch.nn.L1Loss(size_average=False).to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(),# net.classifier.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=gamma)

        dataset_size = len(dataset)

        self.model.train()

        print(f'starting to train...')

        for epoch in range(num_epochs):

            t1 = time.perf_counter()

            epoch_loss = 0
            scheduler.step()

            for data in train_loader:


                current = data['current'].to(self.device)
                next = data['next'].to(self.device)
                delta = data['delta'].to(self.device)

                optimizer.zero_grad()

                output = self.model(current, next)

                loss = criterion(output * 10, delta)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            t2 = time.perf_counter()
            print(f'epoch {epoch}, loss = {epoch_loss:.1f}, avg loss = {epoch_loss / dataset_size:.1f}, duration = {t2 - t1:.1f}')


        torch.save(self.model.state_dict(), 'saved_models/final_model.pth')



def main():
    args = cli.get_args()

    base = args.data_directory

    # base = r'C:\Users\catph\data\kitti_raw\sync\kitti_raw_data\data'

    dataset_list = datasets.get_kitti_datasets(base, (0, 30))

    goturn_model = models.LidarGoturnModel()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(dataset_list, goturn_model, device)

    trainer.train()


if __name__ == '__main__':
    main()
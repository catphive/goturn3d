
import typing
import torch
import datasets

import datasets
import models
import time
import cli

class Validator:

    dataset_list: typing.List[datasets.KittiDataset]
    model: models.LidarGoturnModel

    def __init__(self, dataset_list, model, device):

        model.to(device)

        self.dataset_list = dataset_list
        self.model = model
        self.device = device


    def validate(self, num_epochs = 10, batch_size = 50, learning_rate=1e-5, momentum=0.9, weight_decay=0.0005):

        dataset = torch.utils.data.ConcatDataset(self.dataset_list)

        print(f'dataset size = {len(dataset)}')

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        criterion = torch.nn.L1Loss(size_average=False).to(self.device)

        # optimizer = torch.optim.SGD(self.model.parameters(),# net.classifier.parameters(),
        #                       lr=learning_rate,
        #                       momentum=momentum,
        #                       weight_decay=weight_decay)

        self.model.eval()

        dataset_size = len(dataset)
        total_loss = 0

        with torch.no_grad():
            for data in train_loader:

                current = data['current'].to(self.device)
                next = data['next'].to(self.device)
                delta = data['delta'].to(self.device)

                output = self.model(current, next)

                loss = criterion(output * 10, delta)

                print(f'compare output: {(output*10)[0:1,:]}, delta: {(delta)[0:1,:]}')

                total_loss += loss.item()


        print(f'total loss = {total_loss}, avg loss= {total_loss / dataset_size}')



def main():
    args = cli.get_args()

    base = args.data_directory

    # model = models.LidarGoturnModel()
    model = models.TestModel()

    model.load_state_dict(torch.load('saved_models/final_model.pth'))

    dataset_list = datasets.get_kitti_datasets(base, args.data_sets)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validator = Validator(dataset_list, model, device)

    validator.validate()




if __name__ == '__main__':
    main()


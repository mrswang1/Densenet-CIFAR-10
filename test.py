from get_data_loader import get_data_loader
from model_utils import save_model,load_model
from densenet import Network
import torch
if __name__=='__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    _, _, testloader, _ = get_data_loader()
    model = Network()
    model = model.to(device)
    mode_state_dict = load_model('/data/ZM/wjdata/', 'cifar_densenet_1.model')
    model.load_state_dict(mode_state_dict)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
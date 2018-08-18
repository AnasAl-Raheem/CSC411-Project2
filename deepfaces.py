from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imread

from myalexnet import MyAlexNet
# cropped images location
actors_folder = "cropped_male_rgb227/"
actresses_folder = "cropped_female_rgb227/"
figures_folder = "figures/"

# problemtic images, to be ignored
# discard_list = ["bracco64.jpg", "bracco86.jpg", "bracco101.jpg", "bracco105.jpg",
#                 "gilpin28.jpg", "chenoweth70.jpg", "harmon45.jpg",
#                 "drescher68.jpg", "drescher78.jpg",
#                 "hader10.jpg", "hader53.jpg", "hader80.jpg", "hader95.jpg",
#                 "vartan13.jpg", "vartan14.jpg", "vartan60.jpg", "vartan69.jpg"]

discard_list = ["bracco86.jpg", "chenoweth70.jpg", 
                "drescher68.jpg", "drescher78.jpg",
                "hader10.jpg", "hader95.jpg",
                "vartan13.jpg", "vartan14.jpg", "vartan60.jpg", "vartan69.jpg"]

def get_images(src_dir, names, test_size, validation_size, training_size):
    files = sorted(os.listdir(src_dir))
    act_dict = {i: [j for j in files if j.startswith(i)] for i in names}
    for i in act_dict.keys():
        files_names = act_dict[i]
        files = []
        for j in files_names:
            if not j in discard_list:
                im = imread(src_dir + j)[:,:,:3]
                im = im - np.mean(im.flatten())
                im = im/np.max(np.abs(im.flatten()))
                im = np.rollaxis(im, -1).astype(np.float32)
                im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
                files.append(im_v)
            else:
                print "discard " + j
        np.random.seed(0)
        np.random.shuffle(files)
        np.random.shuffle(files)
        np.random.shuffle(files)
        if i == 'gilpin':
            temp_test_size = 0
            if test_size > 12:
                temp_test_size = test_size
                test_size = 12
        act_dict[i] = [files[0:test_size], files[test_size:test_size + validation_size],
                       files[test_size + validation_size:min(test_size + validation_size + training_size, len(files))]]
        if i == 'gilpin':
            if temp_test_size > 0:
                test_size = temp_test_size
    return act_dict

def part10():
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']

    # retrive data in act_dict
    act_dict = {}
    act_dict.update(get_images(actors_folder, act_names[:3], 20, 10, 70))
    act_dict.update(get_images(actresses_folder, act_names[3:], 20, 10, 70))
    
    alex_model = MyAlexNet()
    alex_model.eval()
    softmax = torch.nn.Softmax()
    
    alex_weights = []
    for i in range(len(act_names)):
        act = []
        for j in range(3):
            set_x = act_dict[act_names[i]][j]
            set_x = torch.cat(set_x)
            act.append(softmax(alex_model.forward(set_x)).data.numpy())
        alex_weights.append(act)
    
    alex_calsses_y = []
    for i in range(len(alex_weights)):
        act = []
        for j in range(3):
            set_x = alex_weights[i][j]
            set_y = np.zeros((set_x.shape[0], 6))
            set_y[:, i] = 1.
            act.append(set_y)
        alex_calsses_y.append(act)
        
    
    training_x, training_y = get_set(alex_weights, alex_calsses_y, set_num = 2)
    validation_x, validation_y = get_set(alex_weights, alex_calsses_y, set_num = 1)
    test_x, test_y = get_set(alex_weights, alex_calsses_y, set_num = 0)
    
    x, y = training_x, training_y
    
    #dim_x = 43264#9216
    dim_x = 9216
    dim_h = 64
    dim_out = 6
    
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        #torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.Linear(dim_h, dim_out),
    )
    results = {}
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for t in range(10000):
        y_pred = model(training_x).data.numpy()
        training_performance = np.mean(np.argmax(y_pred, 1) == training_y.data)
        y_pred = model(validation_x).data.numpy()
        validation_performance = np.mean(np.argmax(y_pred, 1) == validation_y.data)

        results[t] = [training_performance, validation_performance]
        # x, y = get_mini_batch(act_dict, 16)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step
        
        if t % 100 == 0:
            y_pred = model(training_x).data.numpy()
            print "Performance:", np.mean(np.argmax(y_pred, 1) == training_y.data)
            #print "loss:", loss
    
    y_pred = model(training_x).data.numpy()
    print "training performance:%", np.mean(np.argmax(y_pred, 1) == training_y.data)
    
    y_pred = model(validation_x).data.numpy()
    print "validation performance:%", np.mean(np.argmax(y_pred, 1) == validation_y.data)
    
    y_pred = model(test_x).data.numpy()
    print "test performance:%", np.mean(np.argmax(y_pred, 1) == test_y.data)

    learning_curve(results, "part10")
    
# return validation, training, or test set
def get_set(alex_weights, alex_calsses_y, set_num = 2):
    
    set_x = []
    set_y = []
    for i in range(len(alex_weights)):
        set_x.append(alex_weights[i][set_num])
        set_y.append(alex_calsses_y[i][set_num])
    
    set_x = np.vstack(set_x)
    set_y = np.vstack(set_y)
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    set_x = Variable(torch.from_numpy(set_x), requires_grad=False).type(dtype_float)
    set_y = Variable(torch.from_numpy(np.argmax(set_y, 1)), requires_grad=False).type(dtype_long)
    
    return set_x, set_y

def learning_curve(results, name):
    lists = sorted(results.items())
    x, y = zip(*lists)

    plt.figure(int(name[-1]))
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Performance %")
    plt.legend(["Training", "Validation"])
    plt.savefig(figures_folder + name + "_learning_curve.png")

# Main Function
if __name__ == '__main__':
    part10()

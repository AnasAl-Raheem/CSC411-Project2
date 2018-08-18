from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

import os
from scipy.misc import imread
#import shutil

#% matplotlib inline

# get pics?

M = loadmat("mnist_all.mat")

# cropped images location
actors_folder = "cropped_male/"
actresses_folder = "cropped_female/"
figures_folder = "figures/"

# problemtic images, to be ignored
discard_list = ["bracco87.jpg", "chenoweth71.jpg", "drescher69.jpg", "drescher81.jpg",
                "hader11.jpg", "hader95.jpg",
                "vartan13.jpg", "vartan14.jpg", "vartan78.jpg", "vartan116.jpg"]

def get_images(src_dir, names, test_size, validation_size, training_size):
    files = sorted(os.listdir(src_dir))
    act_dict = {i: [j for j in files if j.startswith(i)] for i in names}
    for i in act_dict.keys():
        files_names = act_dict[i]
        files = []
        for j in files_names:
            if not j in discard_list:
                im = imread(src_dir + j, "L")
                # files.append(np.append([1], np.ndarray.flatten(im) / 255.0))
                files.append(np.ndarray.flatten(im) / 255.0)
            else:
                print "discard " + j
        np.random.seed(0)
        np.random.shuffle(files)
        np.random.shuffle(files)
        np.random.shuffle(files)
        act_dict[i] = [files[0:test_size], files[test_size:test_size + validation_size],
                       files[test_size + validation_size:min(test_size + validation_size + training_size, len(files))]]
    return act_dict

def part8_9():
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']

    # retrive data in act_dict
    act_dict = {}
    act_dict.update(get_images(actors_folder, act_names[:3], 20, 10, 70))
    act_dict.update(get_images(actresses_folder, act_names[3:], 20, 10, 70))

    training_x, training_y = get_set(act_dict, 2)
    validation_x, validation_y = get_set(act_dict, 1)
    test_x, test_y = get_set(act_dict, 0)
    
    #x,y = training_x, training_y
    dim_x = 32 * 32
    dim_h = 12
    dim_out = 6

    np.random.seed(0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        #torch.nn.Tanh(),
        torch.nn.Linear(dim_h, dim_out),
    )
    results = {}
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for t in range(3000):
        y_pred = model(training_x).data.numpy()
        training_performance = np.mean(np.argmax(y_pred, 1) == training_y.data)
        y_pred = model(validation_x).data.numpy()
        validation_performance = np.mean(np.argmax(y_pred, 1) == validation_y.data)
        
        results[t] = [training_performance, validation_performance]
        
        x, y = get_mini_batch(act_dict, 16)
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

    learning_curve(results, "part8")
    
    # Part 9, baldwin:
    x = test_x[0:20, :].data.numpy()
    y = np.zeros((x.shape[0], 6))
    y[:, 0] = 1.
    get_weights(model, loss_fn, x, y, 0)
    
    # Part 9, bracco:
    x = test_x[60:80, :].data.numpy()
    y[:, 0] = 0
    y[:, 3] = 1.
    get_weights(model, loss_fn, x, y, 3)
    

def get_weights(model, loss_fn, x, y, indx):
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    x = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
    y = Variable(torch.from_numpy(np.argmax(y, 1)), requires_grad=False).type(dtype_long)
    
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()
    
    act_h = np.where(model[2].weight.grad[indx,:].data.numpy() != 0)[0]

    plt.close('all')
    f, axarr = plt.subplots(1, len(act_h))
    for i in range(len(act_h)): 
        im = model[0].weight.grad[act_h[i]].data.numpy().reshape((32, 32))
        axarr[i].imshow(im, cmap=plt.cm.coolwarm)
    plt.setp([a.get_xticklabels() for a in axarr[:]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:]], visible=False)
    f.tight_layout()
    f.savefig(figures_folder + "part9_" + act_names[indx] + "_weights.png")
    plt.close('all')

def learning_curve(results, name):
    lists = sorted(results.items())
    x, y = zip(*lists)

    plt.figure(int(name[-1]))
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Performance %")
    plt.legend(["Training", "Validation"])
    plt.savefig(figures_folder + name + "_learning_curve.png")

# return validation, training, or test set
def get_set(act_dict, set_num):
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']

    set_act_x = []
    set_act_y = []
    for i in range(6):
        set_x = np.array(act_dict.get(act_names[i])[set_num])
        set_y = np.zeros((set_x.shape[0], 6))
        set_y[:, i] = 1.

        set_act_x.append(set_x)
        set_act_y.append(set_y)

    set_act_x = np.vstack(set_act_x)
    set_act_y = np.vstack(set_act_y)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    set_x = Variable(torch.from_numpy(set_act_x), requires_grad=False).type(dtype_float)
    set_y = Variable(torch.from_numpy(np.argmax(set_act_y, 1)), requires_grad=False).type(dtype_long)
    return set_x, set_y


# return a mini batch of the training set, with mini_batch_size image per actor.
def get_mini_batch(act_dict, mini_batch_size = 16):
    act_names = ['baldwin', 'hader', 'carell', 'bracco', 'gilpin', 'harmon']

    
    # add training data to act
    act = []
    act_y = []
    for i in range(6):
        train_size = len(act_dict.get(act_names[i])[2])
        lower = np.random.randint(0, (train_size - 1) - mini_batch_size)
        act.append(np.array(act_dict.get(act_names[i])[2][lower:lower + mini_batch_size]))
        y = np.zeros((mini_batch_size, 6))
        y[:, i] = 1.
        act_y.append(y)

    x = np.vstack(act)
    y = np.vstack(act_y)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    x = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
    y = Variable(torch.from_numpy(np.argmax(y, 1)), requires_grad=False).type(dtype_long)
    return x, y


# Main Function
if __name__ == '__main__':
    part8_9()
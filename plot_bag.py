import rosbag
import csv
import bagpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bagpy import bagreader


def read_bag_file():
    position_data= []
    bag_file = '2023-02-11-16-31-27.bag'
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=['/mavros/local_position/pose']):
        print(topic)
        position_data.append(msg)
    bag.close()
    print(position_data)


def plot_3d_axes():
    # path = "local_position.csv"
    # data = np.loadtxt(open(path, "rb"))
    # print(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show()
    
    
def bag2csv():
    


if __name__ == '__main__':
    # read_bag_file()
    plot_3d_axes()

# import bagpy
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from bagpy import bagreader
from mpl_toolkits.mplot3d import Axes3D  # 导入库


columns = ["Time",
           "pose.position.x",
           "pose.position.y",
           "pose.position.z"]
columns2 = ["Time", "mode"]
columns3 = ["Time",
            "twist.linear.x",
            "twist.linear.y",
            "twist.linear.z"]
columns4 = ["Time",
            "vector.x",
            "vector.y",
            "vector.z"]
columns5 = ["Time",
            "twist.angular.x",
            "twist.angular.y",
            "twist.angular.z"]
columns6 = ["Time",
            "body_rate.x",
            "body_rate.y",
            "body_rate.z"]


def bag2csv():
    b = bagreader('2023-02-15-21-55-34.bag')

    # LASER_MSG = b.message_by_topic('/mavros/local_position/pose')
    # LASER_MSG
    # df_laser = pd.read_csv(LASER_MSG)
    # df_laser # prints laser data in the form of pandas dataframe

    csvfiles = []
    for t in b.topics:
        data = b.message_by_topic(t)
        csvfiles.append(data)


def plot_states(bag):  # 2023-02-15-21-55-34
    # fig1, ax1 = plt.subplots(4, 3)

    df = pd.read_csv(bag + "/mavros-local_position-pose.csv", usecols=columns)
    dn = pd.read_csv(bag + "/mavgnc-position_setpoint.csv", usecols=columns)

    dm = pd.read_csv("2023-02-15-21-55-34/mavros-state.csv", usecols=columns2)

    set_velo = pd.read_csv(
        bag + "/mavgnc-velocity_setpoint.csv", usecols=columns3)
    lo_velo = pd.read_csv(
        bag + "/mavros-local_position-velocity_local.csv", usecols=columns3)

    set_att = pd.read_csv(bag + "/mavgnc-att_sp_euler.csv", usecols=columns4)
    loc_att = pd.read_csv(bag + "/mavgnc-att_euler.csv", usecols=columns4)

    set_rate = pd.read_csv(
        bag + "/mavros-setpoint_raw-attitude.csv", usecols=columns6)
    loc_rate = pd.read_csv(
        bag + "/mavros-local_position-velocity_local.csv", usecols=columns5)

    df = df.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})

    dn = dn.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})

    set_velo = set_velo.rename(columns={"twist.linear.x": 'vx',
                                        "twist.linear.y": 'vy',
                                        "twist.linear.z": 'vz'})

    lo_velo = lo_velo.rename(columns={"twist.linear.x": 'vx',
                                      "twist.linear.y": 'vy',
                                      "twist.linear.z": 'vz'})

    set_att = set_att.rename(columns={"vector.x": 'qx',
                                      "vector.y": 'qy',
                                      "vector.z": 'qz'})

    loc_att = loc_att.rename(columns={"vector.x": 'qx',
                                      "vector.y": 'qy',
                                      "vector.z": 'qz'})

    set_rate = set_rate.rename(columns={"body_rate.x": 'p',
                                        "body_rate.y": 'q',
                                        "body_rate.z": 'r'})

    loc_rate = loc_rate.rename(columns={"twist.angular.x": 'p',
                                        "twist.angular.y": 'q',
                                        "twist.angular.z": 'r'})

    t = []
    tx = []
    ts = []
    set_velo_t = []
    lo_velo_t = []
    set_att_t = []
    loc_att_t = []
    set_rate_t = []
    loc_rate_t = []

    posx = []
    posy = []
    posz = []
    setx = []
    sety = []
    setz = []

    setvx = []
    setvy = []
    setvz = []
    localvx = []
    localvy = []
    localvz = []

    setqx = []
    setqy = []
    setqz = []
    localqx = []
    localqy = []
    localqz = []

    setp = []
    setq = []
    setr = []
    localp = []
    localq = []
    localr = []

    for i in range(1, len(dm)):
        dm["Time"][i] = dm["Time"][i] - dm["Time"][0]
    dm["Time"][0] = 0
    ##
    for i in range(1, len(df)):
        df.Time[i] = df.Time[i] - df.Time[0]
    df.Time[0] = 0.0

    for i in range(1, len(dn)):
        dn["Time"][i] = dn["Time"][i] - dn["Time"][0]
    dn["Time"][0] = 0
    ##
    for i in range(1, len(set_velo)):
        set_velo["Time"][i] = set_velo["Time"][i] - set_velo["Time"][0]
    set_velo["Time"][0] = 0

    for i in range(1, len(lo_velo)):
        lo_velo["Time"][i] = lo_velo["Time"][i] - lo_velo["Time"][0]
    lo_velo["Time"][0] = 0
    ##
    for i in range(1, len(set_att)):
        set_att["Time"][i] = set_att["Time"][i] - set_att["Time"][0]
    set_att["Time"][0] = 0

    for i in range(1, len(loc_att)):
        loc_att["Time"][i] = loc_att["Time"][i] - loc_att["Time"][0]
    loc_att["Time"][0] = 0
    ##
    for i in range(1, len(set_rate)):
        set_rate["Time"][i] = set_rate["Time"][i] - set_rate["Time"][0]
    set_rate["Time"][0] = 0

    for i in range(1, len(loc_rate)):
        loc_rate["Time"][i] = loc_rate["Time"][i] - loc_rate["Time"][0]
    loc_rate["Time"][0] = 0
    ##
    for i in range(1, len(dm)):
        if dm["mode"][i] == "OFFBOARD":
            t.append(dm.Time[i])
    ########################
    for i in range(1, len(df)):
        if df.Time[i] > t[0]:
            if df.Time[i] < t[-1]:
                tx.append(df.Time[i])
                posx.append(df.x[i])
                posy.append(df.y[i])
                posz.append(df.z[i])

    for i in range(1, len(dn)):
        if dn.Time[i] > t[0]:
            if dn.Time[i] < t[-1]:
                ts.append(dn.Time[i])
                setx.append(dn.x[i])
                sety.append(dn.y[i])
                setz.append(dn.z[i])
    # 3
    for i in range(1, len(set_velo)):
        if set_velo.Time[i] > t[0]:
            if set_velo.Time[i] < t[-1]:
                set_velo_t.append(set_velo.Time[i])
                setvx.append(set_velo.vx[i])
                setvy.append(set_velo.vy[i])
                setvz.append(set_velo.vz[i])

    for i in range(1, len(lo_velo)):
        if lo_velo.Time[i] > t[0]:
            if lo_velo.Time[i] < t[-1]:
                lo_velo_t.append(lo_velo.Time[i])
                localvx.append(lo_velo.vx[i])
                localvy.append(lo_velo.vy[i])
                localvz.append(lo_velo.vz[i])
    ##########################
    for i in range(1, len(set_att)):
        if set_att.Time[i] > t[0]:
            if set_velo.Time[i] < t[-1]:
                set_att_t.append(set_att.Time[i])
                setqx.append(set_att.qx[i])
                setqy.append(set_att.qy[i])
                setqz.append(set_att.qz[i])

    for i in range(1, len(loc_att)):
        if loc_att.Time[i] > t[0]:
            if loc_att.Time[i] < t[-1]:
                loc_att_t.append(loc_att.Time[i])
                localqx.append(loc_att.qx[i])
                localqy.append(loc_att.qy[i])
                localqz.append(loc_att.qz[i])

    ##########################
    for i in range(1, len(set_rate)):
        if set_rate.Time[i] > t[0]:
            if set_rate.Time[i] < t[-1]:
                set_rate_t.append(set_rate.Time[i])
                setp.append(set_rate.p[i])
                setq.append(set_rate.q[i])
                setr.append(set_rate.r[i])

    for i in range(1, len(loc_rate)):
        if loc_rate.Time[i] > t[0]:
            if loc_rate.Time[i] < t[-1]:
                loc_rate_t.append(loc_rate.Time[i])
                localp.append(loc_rate.p[i])
                localq.append(loc_rate.q[i])
                localr.append(loc_rate.r[i])

    tx = tx - tx[0]
    ts = ts - ts[0]
    set_velo_t = set_velo_t - set_velo_t[0]
    lo_velo_t = lo_velo_t - lo_velo_t[0]
    set_att_t = set_att_t - set_att_t[0]
    loc_att_t = loc_att_t - loc_att_t[0]

    #####################
    plt.subplot(4, 3, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    plt.plot(tx, posx, color="blue", label='real_position')
    plt.plot(ts, setx, color="red", label='set_position')
    plt.title("Position_X")
    plt.grid(axis='both')
    plt.ylabel('X [m]')
    plt.legend()

    plt.subplot(4, 3, 2)
    plt.plot(tx, posy, color="blue", label='real_position')
    plt.plot(ts, sety, color="red", label='set_position')
    plt.title("Position_Y")
    plt.grid(axis='both')
    plt.ylabel('Y [m]')
    plt.legend()

    plt.subplot(4, 3, 3)
    plt.plot(tx, posz, color="blue", label='real_position')
    plt.plot(ts, setz, color="red", label='set_position')
    plt.title("Position_Z")
    plt.grid(axis='both')
    plt.ylabel('Z [m]')
    plt.legend()

    ##################
    plt.subplot(4, 3, 4)
    plt.plot(lo_velo_t, localvx, color="blue", label='real_position')
    plt.plot(set_velo_t, setvx, color="red", label='set_position')
    plt.title("Velocity_X")
    plt.grid(axis='both')
    plt.ylabel('Vx [m/s]')
    plt.legend()

    plt.subplot(4, 3, 5)
    plt.plot(lo_velo_t, localvy, color="blue", label='real_position')
    plt.plot(set_velo_t, setvy, color="red", label='set_position')
    plt.title("Velocity_Y")
    plt.grid(axis='both')
    plt.ylabel('Vy [m/s]')
    plt.legend()

    plt.subplot(4, 3, 6)
    plt.plot(lo_velo_t, localvz, color="blue", label='real_position')
    plt.plot(set_velo_t, setvz, color="red", label='set_position')
    plt.title("Velocity_Z")
    plt.grid(axis='both')
    plt.ylabel('Vz [m/s]')
    plt.legend()

    ########################
    plt.subplot(4, 3, 7)
    plt.plot(loc_att_t, localqx, color="blue", label='real_position')
    plt.plot(set_att_t, setqx, color="red", label='set_position')
    plt.title("Attitude_X")
    plt.grid(axis='both')
    plt.ylabel('Roll [deg]')
    plt.legend()

    plt.subplot(4, 3, 8)
    plt.plot(loc_att_t, localqy, color="blue", label='real_position')
    plt.plot(set_att_t, setqy, color="red", label='set_position')
    plt.title("Attitude_Y")
    plt.grid(axis='both')
    plt.ylabel('Pitch [deg]')
    plt.legend()

    plt.subplot(4, 3, 9)
    plt.plot(loc_att_t, localqz, color="blue", label='real_position')
    plt.plot(set_att_t, setqz, color="red", label='set_position')
    plt.title("Attitude_Z")
    plt.grid(axis='both')
    plt.ylabel('Yaw [deg]')
    plt.legend()

    #######################
    plt.subplot(4, 3, 10)
    plt.plot(loc_rate_t, localp, color="blue", label='real_position')
    plt.plot(set_rate_t, setp, color="red", label='set_position')
    plt.title("Body_Rate_X")
    plt.grid(axis='both')
    plt.ylabel('P  [rad/s]')
    plt.xlabel('t [s]')
    plt.legend()

    plt.subplot(4, 3, 11)
    plt.plot(loc_rate_t, localq, color="blue", label='real_position')
    plt.plot(set_rate_t, setq, color="red", label='set_position')
    plt.title("Body_Rate_Y")
    plt.grid(axis='both')
    plt.ylabel('Q [rad/s]')
    plt.xlabel('t [s]')
    plt.legend()

    plt.subplot(4, 3, 12)
    plt.plot(loc_rate_t, localr, color="blue", label='real_position')
    plt.plot(set_rate_t, setr, color="red", label='set_position')
    plt.title("Body_Rate_Z")
    plt.grid(axis='both')
    plt.ylabel('R [rad/s]')
    plt.xlabel('t [s]')
    plt.legend()

    plt.suptitle("Quadrotor Real Data")
    plt.show()


def csv2plt():
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    columns = ["Time", "pose.position.x", "pose.position.y", "pose.position.z"]
    columns2 = ["Time", "mode"]
    df = pd.read_csv(
        "2023-02-11-16-31-27/mavros-local_position-pose.csv", usecols=columns)
    dm = pd.read_csv("2023-02-11-16-31-27/mavros-state.csv", usecols=columns2)
    df = df.rename(columns={"pose.position.x": 'x',
                   "pose.position.y": 'y', "pose.position.z": 'z'})
    t = []
    posx = []
    tx = []

    for i in range(1, len(df)):
        df.Time[i] = df.Time[i] - df.Time[0]
    df.Time[0] = 0.0

    for i in range(1, len(dm)):
        dm["Time"][i] = dm["Time"][i] - dm["Time"][0]
    dm["Time"][0] = 0
    # print(dm["Time"])

    for i in range(1, len(dm)):
        if dm["mode"][i] == "OFFBOARD":
            t.append(dm.Time[i])

    for i in range(1, len(df)):
        if df.Time[i] > t[0]:
            if df.Time[i] < t[-1]:
                tx.append(df.Time[i])
                posx.append(df.x[i])

    tx = tx - tx[0]

    # print(tx)
    plt.plot(tx, posx)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()


def plot():
    time = 0
    t = []
    x = []
    y = []
    z = []
    fig1, ax1 = plt.subplots(4, 3)
    with open('2023-02-11-16-31-27/mavros-local_position-pose.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        next(plots)
        for row in plots:
            t.append(time)
            x.append(row[5])
            y.append(row[6])
            z.append(row[7])
            time = time + 1
    # print(x)
    ax1[0, 0].plot(t, x, label='real')
    # ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
    ax1[0, 0].set_ylabel('x')
    ax1[0, 0].set_xlabel('t')
    ax1[0, 1].plot(t, y, label='real')
    ax1[0, 1].set_ylabel('y')
    ax1[0, 1].set_xlabel('t')
    # ax1[0,1].plot(self.time,self.position_cmd[:,1])
    ax1[0, 2].plot(t, z, label='real')
    # ax1[0,2].plot(self.time,self.position_cmd[:,2])
    ax1[0, 2].set_ylabel('z')
    ax1[0, 2].set_xlabel('t')
    ax1[0, 0].legend()
    plt.show()


def plot_3d(bag1, bag2):  # 2023-02-11-16-31-27 2023-02-11-16-31-28
    columns = ["Time",
               "pose.position.x",
               "pose.position.y",
               "pose.position.z"]
    columns2 = ["Time", "mode"]

    df = pd.read_csv(
        bag1 + "/mavros-local_position-pose.csv", usecols=columns)
    dn = pd.read_csv(
        bag2 + "/mavros-local_position-pose.csv", usecols=columns)
    dm = pd.read_csv(bag1 + "/mavros-state.csv", usecols=columns2)
    dq = pd.read_csv(bag2 + "/mavros-state.csv", usecols=columns2)

    df = df.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    dn = dn.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })

    for i in range(1, len(df)):
        df.Time[i] = df.Time[i] - df.Time[0]
    df.Time[0] = 0.0

    for i in range(1, len(dm)):
        dm["Time"][i] = dm["Time"][i] - dm["Time"][0]
    dm["Time"][0] = 0
    # print(dm["Time"])

    for i in range(1, len(dq)):
        dq["Time"][i] = dq["Time"][i] - dq["Time"][0]
    dq["Time"][0] = 0

    for i in range(1, len(dn)):
        dn["Time"][i] = dn["Time"][i] - dn["Time"][0]
    dn["Time"][0] = 0

    t = []
    t2 = []
    tx = []
    ts = []
    posx = []
    posy = []
    posz = []
    setx = []
    sety = []
    setz = []

    for i in range(1, len(dm)):
        if dm["mode"][i] == "OFFBOARD":
            t.append(dm.Time[i])
    for i in range(1, len(dq)):
        if dq["mode"][i] == "OFFBOARD":
            t2.append(dq.Time[i])

    for i in range(1, len(df)):
        if df.Time[i] > t[0]:
            if df.Time[i] < t[-1]:
                tx.append(df.Time[i])
                posx.append(df.x[i])
                posy.append(df.y[i])
                posz.append(df.z[i])

    for i in range(1, len(dn)):
        if dn.Time[i] > t2[0]:
            if dn.Time[i] < t2[-1]:
                tx.append(dn.Time[i])
                setx.append(dn.x[i])
                sety.append(dn.y[i])
                setz.append(dn.z[i])

    plt.subplot(projection='3d')
    # plt.subplot()
    plt.plot(posx, posy, posz, color="blue", label="UAV0")
    plt.plot(setx, sety, setz, color="red", label="UAV1")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot()
    # plot_states("2023-02-15-21-55-34")
    # plot_states("2023-02-15-21-54-18")
    plot_3d("2023-02-11-16-31-27", "2023-02-11-16-31-28")

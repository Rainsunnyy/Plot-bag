# import bagpy
import csv
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from bagpy import bagreader
from mpl_toolkits.mplot3d import Axes3D  # 导入库
from matplotlib.backends.backend_pdf import PdfPages

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


def bag2csv(bag):
    b = bagreader(bag)
    
    csvfiles = []
    for t in b.topics:
        data = b.message_by_topic(t)
        csvfiles.append(data)


def plot_states(bag, bag2): 
    df = pd.read_csv(bag + "/mavros-local_position-pose.csv", usecols=columns)
    dn = pd.read_csv(bag + "/mavgnc-position_setpoint.csv", usecols=columns)

    dm = pd.read_csv(bag + "/mavros-state.csv", usecols=columns2)

    df2 = pd.read_csv(bag2 + "/mavros-local_position-pose.csv", usecols=columns)
    dn2 = pd.read_csv(bag2 + "/mavgnc-position_setpoint.csv", usecols=columns)

    dm2 = pd.read_csv(bag2 + "/mavros-state.csv", usecols=columns2)

    df = df.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})

    dn = dn.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})
    df2= df2.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})

    dn2 = dn2.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z'})

    t = []
    t2 = []
    tx = []
    ts = []
    t2 = []
    tx2 = []
    ts2 = []

    posx = []
    posy = []
    posz = []
    setx = []
    sety = []
    setz = []
    posx2 = []
    posy2 = []
    posz2 = []
    setx2 = []
    sety2 = []
    setz2 = []

    for i in range(1, len(dm)):
        dm["Time"][i] = dm["Time"][i] - dm["Time"][0]
    dm["Time"][0] = 0
    
    for i in range(1, len(dm2)):
        dm2["Time"][i] = dm2["Time"][i] - dm2["Time"][0]
    dm2["Time"][0] = 0
    ##################################
    for i in range(1, len(df)):
        df.Time[i] = df.Time[i] - df.Time[0]
    df.Time[0] = 0.0

    for i in range(1, len(dn)):
        dn["Time"][i] = dn["Time"][i] - dn["Time"][0]
    dn["Time"][0] = 0
    
    for i in range(1, len(df2)):
        df2.Time[i] = df2.Time[i] - df2.Time[0]
    df2.Time[0] = 0.0

    for i in range(1, len(dn2)):
        dn2["Time"][i] = dn2["Time"][i] - dn2["Time"][0]
    dn2["Time"][0] = 0
    ##################################
    for i in range(1, len(dm)):
        if dm["mode"][i] == "OFFBOARD":
            t.append(dm.Time[i])
    for i in range(1, len(dm2)):
        if dm2["mode"][i] == "OFFBOARD":
            t2.append(dm2.Time[i])
    ##################################
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

    for i in range(1, len(df2)):
        if df2.Time[i] > t[0]:
            if df2.Time[i] < t[-1]:
                tx2.append(df2.Time[i])
                posx2.append(df2.x[i])
                posy2.append(df2.y[i])
                posz2.append(df2.z[i])

    for i in range(1, len(dn2)):
        if dn2.Time[i] > t[0]:
            if dn2.Time[i] < t[-1]:
                ts2.append(dn.Time[i])
                setx2.append(dn2.x[i])
                sety2.append(dn2.y[i])
                setz2.append(dn2.z[i])
    
    fig1, ax1 = plt.subplots(3, 1)
    sns.set_style("whitegrid")
    sns.set()
    
    # tx2 = tx2 - tx2[0]
    # ts2 = ts2 - ts2[0]
    
    data_plot = pd.DataFrame({"t":tx, "x":posx})
    data = pd.DataFrame({"t":ts, "x":setx})
    data_y = pd.DataFrame({"t":tx, "y":posy})
    datay = pd.DataFrame({"t":ts, "y":sety})
    data_z = pd.DataFrame({"t":tx, "z":posz})
    dataz = pd.DataFrame({"t":ts, "z":setz})

    data_plot2 = pd.DataFrame({"t":tx2, "x":posx2})
    data2 = pd.DataFrame({"t":ts2, "x":setx2})
    data_y2 = pd.DataFrame({"t":tx2, "y":posy2})
    datay2 = pd.DataFrame({"t":ts2, "y":sety2})
    data_z2 = pd.DataFrame({"t":tx2, "z":posz2})
    dataz2 = pd.DataFrame({"t":ts2, "z":setz2})
    
    
    
    sns.lineplot(x = "t", y = "x", data=data_plot, label='UAV0', color= "dodgerblue", ax=ax1[0])
    sns.lineplot(x = "t", y = "x", data=data, linestyle='--', label='reference', color= "dodgerblue", ax=ax1[0])
    sns.lineplot(x = "t", y = "x", data=data_plot2, label='UAV1', color= "tomato", ax=ax1[0])
    sns.lineplot(x = "t", y = "x", data=data2, linestyle='--', label='reference', color= "tomato", ax=ax1[0])
    
    sns.lineplot(x = "t", y = "y", data=data_y, label='UAV0', color= "dodgerblue", ax=ax1[1])
    sns.lineplot(x = "t", y = "y", data=datay, linestyle='--', label='reference', color= "dodgerblue", ax=ax1[1])
    sns.lineplot(x = "t", y = "y", data=data_y2, label='UAV1', color= "tomato", ax=ax1[1])
    sns.lineplot(x = "t", y = "y", data=datay2, linestyle='--', label='reference', color= "tomato", ax=ax1[1])
        
    sns.lineplot(x = "t", y = "z", data=data_z, label='UAV0', color= "dodgerblue", ax=ax1[2])
    sns.lineplot(x = "t", y = "z", data=dataz, linestyle='--', label='reference', color= "dodgerblue", ax=ax1[2])
    sns.lineplot(x = "t", y = "z", data=data_z2, label='UAV1', color= "tomato", ax=ax1[2])
    sns.lineplot(x = "t", y = "z", data=dataz2, linestyle='--', label='reference', color= "tomato", ax=ax1[2])
    # plt.xlim(11, 15.8)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
            
    # fig1, ax1 = plt.subplots(2, 3)
    # plt.subplots_adjust(top=0.9, bottom= 0.6, left=0.035, right=0.985, hspace=0.5, wspace=0.12)
    # ax1[0, 0].plot(tx, posx, color="tomato", label='real_position')
    # ax1[0, 0].plot(ts, setx, color="dodgerblue", label='set_position')
    # ax1[0, 0].set_xlim(10.5, 16)
    # ax1[0, 0].set_ylabel('X [m]')
    # ax1[0, 0].legend()
    
    # ax1[0, 1].plot(tx, posy, color="tomato", label='real_position')
    # ax1[0, 1].plot(ts, sety, color="dodgerblue", label='set_position')
    # ax1[0, 1].set_xlim(10.5, 16)
    # ax1[0, 1].set_ylabel('Y [m]')
    # ax1[0, 1].legend()
    # ax1[0, 1].set_title("UAV0")
    # ax1[1, 1].set_title("UAV1")
    # plt.suptitle("Quadrotor Real Data")
    
    # plt.show()


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

    plt.style.use('classic')
    # print(len(posx))
    del setx[165: 570]
    del sety[165: 570]
    del setz[165: 570]

    del posx[300: 540]
    del posy[300: 540]
    del posz[300: 540]
    del posx[0: 110]
    del posy[0: 110]
    del posz[0: 110]
    
    # plt.subplot(projection='3d')
    # # plt.subplot()
    # plt.xlim((-2, 2))
    # plt.ylim((-2, 2))
    # plt.zlim((-1, 1))
    
    # plt.plot(posx, posy, posz, color="blue", label="UAV0")
    # # plt.plot(setx, sety, setz, color="red", label="UAV1")
    # plt.xlabel('X[m]')
    # plt.ylabel('Y[m]')
    # plt.title('Trajectory')
    # plt.legend()
    # plt.show()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=13, azim=60)
    ax.plot3D(posx, posy, posz, 'tomato', linewidth=4, label='UAV0')
    ax.plot3D(setx, sety, setz,'dodgerblue', linewidth=4, label='UAV1')
    ax.text3D(0, -0.25, 2, "Gate1", fontsize=15)
    ax.text3D(1.2, 0, 2, "Gate2", fontsize=15)
    ax.plot3D([-0.3, -0.3, -0.3, -0.3, -0.3], [-0.5, 0, 0, -0.5, -0.5], [1, 1, 2, 2, 1], 'orange', linewidth=8)
    ax.plot3D([-0.3, -0.3, -0.3, -0.3, -0.3], [-0.5, 0, 0, -0.5, -0.5], [1, 1, 2, 2, 1], 'orange', linewidth=8)
    ax.plot3D([0.8, 1.5, 1.5, 0.8, 0.8], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'orange', linewidth=8)
    ax.plot3D([0.8, 1.5, 1.5, 0.8, 0.8], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'orange', linewidth=8)


    x = np.linspace(-2.3, 2.3, 9)
    y = np.linspace(-1.3, 1.3, 9)
    z = np.linspace(0, 2.5, 9)
    X, Y = np.meshgrid(x, y)
    T, Z = np.meshgrid(y, z)
    ax.plot_surface(X, Y, Z = X * 0 + 0, color='white', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y, Z = X * 0 + 2.5, color='white', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 - 1.3, Z = Z, color='white', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 + 1.3, Z = Z, color='white', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 - 2.3, Y = T, Z = Z, color='white', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 + 2.3, Y = T, Z = Z, color='white', alpha=0.1, edgecolors='white')

    x2 = np.linspace(-2, 2, 9)
    y2 = np.linspace(-1, 1, 9)
    z2 = np.linspace(1, 2, 9)
    X2, Y2 = np.meshgrid(x2, y2)
    T2, Z2 = np.meshgrid(y2, z2)
    
    ax.plot_surface(X = X2, Y = Y2, Z = X2 * 0 + 1, color='navajowhite', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X2, Y = Y2, Z = X2 * 0 + 2, color='navajowhite', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X2, Y = X2 * 0 - 1, Z = Z2, color='navajowhite', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X2, Y = X2 * 0 + 1, Z = Z2, color='navajowhite', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X2 * 0 - 2, Y = T2, Z = Z2, color='navajowhite', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X2 * 0 + 2, Y = T2, Z = Z2, color='navajowhite', alpha=0.1, edgecolors='white')

    ax.scatter3D(posx[0], posy[0], posz[0], linewidth=2)
    ax.scatter3D(setx[0], sety[0], setz[0], linewidth=2)
    ax.text3D(posx[0]+0.2, posy[0]+0.2, posz[0]+0.2, "start", fontsize=20)
    ax.text3D(setx[0], sety[0], setz[0], "start", fontsize=20)
    ax.scatter3D(posx[len(posx)-1], posy[len(posy)-1], posz[len(posz)-1], linewidth=2)
    ax.scatter3D(setx[len(setx)-1], sety[len(sety)-1], setz[len(setz)-1], linewidth=2)
    ax.text3D(posx[len(posx)-1], posy[len(posy)-1], posz[len(posz)-1], "end", fontsize=20)
    ax.text3D(setx[len(setx)-1], sety[len(sety)-1], setz[len(setz)-1], "end", fontsize=20)
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1, 1))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1, 1))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1, 1))
    
    # ax.annotate('text', xy=(posx[0], posy[0]), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.set_xlim(-2.3, 2.3)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0, 2.5)
    ax.set_zlabel("Z [m]")
    
    ax.set_title('Two UAVs Trajectories')
    
    ax.legend()
    plt.gca().set_box_aspect((3, 2, 0.5))
    # plt.axis('off')

    plt.show()
    

def plot_sim(bag):
    columns = ["Time",
               "pose.position.x",
               "pose.position.y",
               "pose.position.z"]

    df = pd.read_csv(
        bag + "/offb1_node-fly_point.csv", usecols=columns)

    d2 = pd.read_csv(
        bag + "/offb2_node-fly_point.csv", usecols=columns)
    d3 = pd.read_csv(
        bag + "/offb3_node-fly_point.csv", usecols=columns)

    df = df.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d2 = d2.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d3 = d3.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })

    posx = []
    posy = []
    posz = []

    pos2x = []
    pos2y = []
    pos2z = []
    
    pos3x = []
    pos3y = []
    pos3z = []
    
    for i in range(1, len(df)):
        posx.append(df.x[i])
        posy.append(df.y[i])
        posz.append(df.z[i])

    for i in range(1, len(d2)):
        pos2x.append(d2.x[i])
        pos2y.append(d2.y[i])
        pos2z.append(d2.z[i])

    for i in range(1, len(d3)):
        pos3x.append(d3.x[i])
        pos3y.append(d3.y[i])
        pos3z.append(d3.z[i])

    # plt.style.use('classic')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=13, azim=60)

    ax.set_xlim3d([10, 40])
    ax.set_ylim3d([10, 40])
    ax.set_zlim3d([1, 7])
    
    # print(len(posx))
    del posx[1280:1401]
    del posy[1280:1401]
    del posz[1280:1401]

    del pos2x[1280:1401]
    del pos2y[1280:1401]
    del pos2z[1280:1401]

    del pos3x[1280:1401]
    del pos3y[1280:1401]
    del pos3z[1280:1401]
    
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.8, 1.8, 0.5, 1]))
    ax.plot3D(posx, posy, posz, 'red', linewidth=3, label='UAV1', alpha=0.8)
    ax.plot3D(pos2x, pos2y, pos2z, 'steelblue', linewidth=3, label='UAV2', alpha=0.8)# dodgerbluetomato
    ax.plot3D(pos3x, pos3y, pos3z, 'gray', linewidth=3, label='UAV3', alpha=0.8)


    x = np.linspace(10, 45, 2)
    y = np.linspace(10, 41, 2)
    z = np.linspace(1, 7, 2)
    X, Y = np.meshgrid(x, y)
    T, Z = np.meshgrid(y, z)
    ax.plot_surface(X, Y, Z = X * 0 + 1, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y, Z = X * 0 + 7, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 + 10, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 + 41, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 + 10, Y = T, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 + 45, Y = T, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    
    # ax.text3D(0, 0.2, 1.4, "Gate1")
    # ax.text3D(1.2, 0, 1.4, "Gate2")
    # ax.plot3D([33, 37, 37, 33, 33], [30, 30, 30, 30, 30], [3, 3, 5, 5, 3], 'darkred', linewidth=8)
    # ax.plot3D([33, 37, 37, 33, 33], [30, 30, 30, 30, 30], [3, 3, 5, 5, 3], 'darkred', linewidth=8)
    # ax.plot3D([0.5, 1.5, 1.5, 0.5, 0.5], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'darkred', linewidth=8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
 
    plt.gca().set_box_aspect((5, 4, 2))
    ax.set_title('Three UAVs Trajectories Simulation')
    ax.legend()
    ax.axis('off')
    plt.show()


def plot_sim_five(bag):
    columns = ["Time",
               "pose.position.x",
               "pose.position.y",
               "pose.position.z"]

    df = pd.read_csv(
        bag + "/offb1_node-fly_point.csv", usecols=columns)

    d2 = pd.read_csv(
        bag + "/offb2_node-fly_point.csv", usecols=columns)
    d3 = pd.read_csv(
        bag + "/offb3_node-fly_point.csv", usecols=columns)
    d4 = pd.read_csv(
        bag + "/offb4_node-fly_point.csv", usecols=columns)
    d5 = pd.read_csv(
        bag + "/offb5_node-fly_point.csv", usecols=columns)

    df = df.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d2 = d2.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d3 = d3.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d4 = d4.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })
    d5 = d5.rename(columns={"pose.position.x": 'x',
                            "pose.position.y": 'y',
                            "pose.position.z": 'z', })

    posx = []
    posy = []
    posz = []

    pos2x = []
    pos2y = []
    pos2z = []
    
    pos3x = []
    pos3y = []
    pos3z = []

    pos4x = []
    pos4y = []
    pos4z = []
    
    pos5x = []
    pos5y = []
    pos5z = []
    
    for i in range(1, len(df)):
        posx.append(df.x[i])
        posy.append(df.y[i])
        posz.append(df.z[i])

    for i in range(1, len(d2)):
        pos2x.append(d2.x[i])
        pos2y.append(d2.y[i])
        pos2z.append(d2.z[i])

    for i in range(1, len(d3)):
        pos3x.append(d3.x[i])
        pos3y.append(d3.y[i])
        pos3z.append(d3.z[i])

    for i in range(1, len(d4)):
        pos4x.append(d4.x[i])
        pos4y.append(d4.y[i])
        pos4z.append(d4.z[i])

    for i in range(1, len(d5)):
        pos5x.append(d5.x[i])
        pos5y.append(d5.y[i])
        pos5z.append(d5.z[i])

    # plt.style.use('classic')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=13, azim=60)

    ax.set_xlim3d([10, 40])
    ax.set_ylim3d([10, 40])
    ax.set_zlim3d([1, 7])
    
    # print(len(posx))
    # del posx[1280:1401]
    # del posy[1280:1401]
    # del posz[1280:1401]

    # del pos2x[1280:1401]
    # del pos2y[1280:1401]
    # del pos2z[1280:1401]

    # del pos3x[1280:1401]
    # del pos3y[1280:1401]
    # del pos3z[1280:1401]
    
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.8, 1.8, 0.5, 1]))
    ax.plot3D(posx, posy, posz, 'red', linewidth=3, label='UAV1', alpha=0.8)
    ax.plot3D(pos2x, pos2y, pos2z, 'steelblue', linewidth=3, label='UAV2', alpha=0.8)# dodgerbluetomato
    ax.plot3D(pos3x, pos3y, pos3z, 'gray', linewidth=3, label='UAV3', alpha=0.8)
    ax.plot3D(pos4x, pos4y, pos4z, 'yellow', linewidth=3, label='UAV4', alpha=0.8)
    ax.plot3D(pos5x, pos5y, pos5z, 'green', linewidth=3, label='UAV5', alpha=0.8)# dodgerbluetomato


    x = np.linspace(10, 45, 2)
    y = np.linspace(10, 41, 2)
    z = np.linspace(1, 7, 2)
    X, Y = np.meshgrid(x, y)
    T, Z = np.meshgrid(y, z)
    ax.plot_surface(X, Y, Z = X * 0 + 1, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y, Z = X * 0 + 7, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 + 10, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X, Y = X * 0 + 41, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 + 10, Y = T, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    ax.plot_surface(X = X * 0 + 45, Y = T, Z = Z, color='lightblue', alpha=0.1, edgecolors='white')
    
    # ax.text3D(0, 0.2, 1.4, "Gate1")
    # ax.text3D(1.2, 0, 1.4, "Gate2")
    # ax.plot3D([33, 37, 37, 33, 33], [30, 30, 30, 30, 30], [3, 3, 5, 5, 3], 'darkred', linewidth=8)
    # ax.plot3D([33, 37, 37, 33, 33], [30, 30, 30, 30, 30], [3, 3, 5, 5, 3], 'darkred', linewidth=8)
    # ax.plot3D([0.5, 1.5, 1.5, 0.5, 0.5], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'darkred', linewidth=8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
 
    plt.gca().set_box_aspect((5, 4, 2))
    ax.set_title('Three UAVs Trajectories Simulation')
    ax.legend()
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    # bag2csv('bag_no_gate.bag')
    # plot_states("2023-02-26-15-41-45", "2023-02-26-15-37-49")
    # plot_states("2023-02-15-21-54-18")
    # plot_3d("2023-02-26-15-41-45", "2023-02-26-15-37-49")
    # plot_sim_five("bag_no_gate")

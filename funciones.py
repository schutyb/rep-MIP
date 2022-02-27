def histogram_line(Ro, x_c, y_c, N, G_filt, S_filt, img_mean, ro, print_fractions=True, plot_histogram=True):
    x1 = np.linspace(start=-1, stop=1, num=500)
    y_positive = lambda x1: np.sqrt(1 - x1 ** 2)
    y_negative = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    y_positive2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    y_negative2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)

    fig = plt.figure(5)
    ax = fig.add_subplot()
    plt.hist2d(x_c, y_c, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-0.5, 0.5], [0, 1]])
    plt.title('Phasor - components determination')
    plt.xlabel('G')
    plt.ylabel('S')

    plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
    plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
    plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
    plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')

    p = plt.ginput(2, timeout=False)

    ax.annotate('Componente A', xy=(p[0][0], p[0][1]),
                xytext=(p[0][0] + 0.25 * abs(p[0][0]), p[0][1] + 0.25 * abs(p[0][1])),
                arrowprops=dict(facecolor='black', arrowstyle='simple'))

    ax.annotate('Componente B', xy=(p[1][0], p[1][1]),
                xytext=(p[1][0] + 0.25 * abs(p[1][0]), p[1][1] + 0.25 * abs(p[1][1])),
                arrowprops=dict(facecolor='black', arrowstyle='simple'))

    plt.plot((p[0][0], p[1][0]), (p[0][1], p[1][1]), 'k')

    circle1 = plt.Circle(p[0], radius=Ro, color='k', fill=False)
    fig.gca().add_artist(circle1)
    circle2 = plt.Circle(p[1], radius=Ro, color='k', fill=False)
    fig.gca().add_artist(circle2)

    circle1 = plt.Circle(p[0], radius=Ro / 10, color='k')
    fig.gca().add_artist(circle1)
    circle2 = plt.Circle(p[1], radius=Ro / 10, color='k')
    fig.gca().add_artist(circle2)

    "x and y are the G and S coordinates"
    x = np.linspace(min(p[0][0], p[1][0]), max(p[0][0], p[1][0]), N)
    y = p[0][1] + ((p[0][1] - p[1][1]) / (p[0][0] - p[1][0])) * (x - p[0][0])

    a = np.array([[p[0][0], p[1][0]], [p[0][1], p[1][1]]])
    mf = np.zeros([len(x), 2])

    for i in range(0, len(x)):
        gs = np.array([x[i], y[i]])
        f = np.linalg.solve(a, gs)
        mf[i][0] = round(f[0], 2)
        mf[i][1] = round(f[1], 2)

    fx = np.linspace(0, 1, N) * 100

    """
    calculate the amount of pixels related to a point in the segment
    calculate the distance between x and x_c the minimal distance means
    that we have found the G coordinate, the same for S
    """
    hist_p = np.zeros(N)
    for ni in range(0, N):

        """
        create a matrix to see if a pixels is into the circle, using circle equation
        so the negative values of Mi means that the pixel belong to the circle
        """
        m1 = (G_filt - x[ni]) ** 2 + (S_filt - y[ni]) ** 2 - ro ** 2
        aux1 = np.zeros([len(img_mean), len(img_mean[0])])

        for i in range(0, len(img_mean)):
            for j in range(0, len(img_mean[0])):
                if img_mean[i][j] > 20:
                    if m1[i][j] < 0:
                        aux1[i][j] = 1

        indices = np.where(aux1 == 1)
        hist_p[ni] = len(indices[0])

    if print_fractions:
        print('Componente A  \t Componente B')
        for i in range(0, len(x)):
            print(mf[i][0], '\t\t', mf[i][1])

    if plot_histogram:
        plt.figure(6)
        plt.plot(fx, hist_p)
        plt.title('pixel histogram')
        plt.show()

    return hist_p


def interactive(img_mean, hist_img, bins, G_filt, S_filt, Ro):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor, Button
    from matplotlib import colors

    fig1, ax = plt.subplots()
    plt.bar(bins[0: len(bins) - 1], hist_img, width=1)
    ax.set_yscale("log")
    cursor = Cursor(ax, horizOn=True, vertOn=True, color='darkgoldenrod')
    plt.title('Average image histogram')
    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x_c = []
    y_c = []

    """store the coordinate to plot in the phasor"""
    for i in range(0, len(G_filt)):
        for j in range(0, len(G_filt[0])):
            if img_mean[i][j] > ic:
                x_c.append(G_filt[i][j])
                y_c.append(S_filt[i][j])

    # built the figure inner and outer circle and the 45 degrees lines in the plot
    x1 = np.linspace(start=-1, stop=1, num=500)
    y_positive = lambda x1: np.sqrt(1 - x1 ** 2)
    y_negative = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    y_positive2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    y_negative2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)

    fig = plt.figure(2)
    plt.hist2d(x_c, y_c, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    plt.title('Phasor')
    plt.xlabel('G')
    plt.ylabel('S')

    plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
    plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
    plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
    plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')

    centro = plt.ginput(3, timeout=0)

    circle1 = plt.Circle(centro[0], radius=Ro, color='r', fill=False)
    fig.gca().add_artist(circle1)
    circle2 = plt.Circle(centro[1], radius=Ro, color='g', fill=False)
    fig.gca().add_artist(circle2)
    circle3 = plt.Circle(centro[2], radius=Ro, color='b', fill=False)
    fig.gca().add_artist(circle3)

    Go1 = centro[0][0]
    So1 = centro[0][1]
    Go2 = centro[1][0]
    So2 = centro[1][1]
    Go3 = centro[2][0]
    So3 = centro[2][1]

    Gn = G_filt
    Sn = S_filt

    """create a matrix to see if a pixels is into the circle, using circle equation
    so the negative values of Mi means that the pixel belong to the circle"""
    M1 = (Gn - Go1) ** 2 + (Sn - So1) ** 2 - Ro ** 2
    M2 = (Gn - Go2) ** 2 + (Sn - So2) ** 2 - Ro ** 2
    M3 = (Gn - Go3) ** 2 + (Sn - So3) ** 2 - Ro ** 2

    aux1 = np.zeros([len(img_mean), len(img_mean[0])])
    aux2 = np.zeros([len(img_mean), len(img_mean[0])])
    aux3 = np.zeros([len(img_mean), len(img_mean[0])])
    for i in range(0, len(img_mean)):
        for j in range(0, len(img_mean[0])):
            if img_mean[i][j] > ic:
                if M1[i][j] < 0:
                    aux1[i][j] = 1
                if M2[i][j] < 0:
                    aux2[i][j] = 1
                if M3[i][j] < 0:
                    aux3[i][j] = 1

    img_new = np.copy(img_mean)

    indices1 = np.where(aux1 == 1)
    indices2 = np.where(aux2 == 1)
    indices3 = np.where(aux3 == 1)

    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))

    # Set the colors
    rgba[indices1[0], indices1[1], :3] = 1, 0, 0  # blue
    rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
    rgba[indices3[0], indices3[1], :3] = 0, 0, 1  # red

    plt.figure(3)
    plt.imshow(img_mean, cmap='seismic')
    plt.title('Average intensity image')

    plt.figure(4)
    plt.imshow(rgba)
    plt.title('Pseudocolor image')
    plt.show()

    return rgba, x_c, y_c


def interactive2(img_mean, x_c, y_c, Ro, g, s, ic):
    # built the figure inner and outer circle and the 45 degrees lines in the plot
    x1 = np.linspace(start=-1, stop=1, num=500)
    y_positive = lambda x1: np.sqrt(1 - x1 ** 2)
    y_negative = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    y_positive2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    y_negative2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)

    fig = plt.figure(1)
    plt.hist2d(x_c, y_c, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    plt.title('Phasor')
    plt.xlabel('G')
    plt.ylabel('S')

    plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
    plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
    plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
    plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')

    centro = plt.ginput(3, timeout=0)

    circle1 = plt.Circle(centro[0], radius=Ro, color='r', fill=False)
    fig.gca().add_artist(circle1)
    circle2 = plt.Circle(centro[1], radius=Ro, color='g', fill=False)
    fig.gca().add_artist(circle2)
    circle3 = plt.Circle(centro[2], radius=Ro, color='b', fill=False)
    fig.gca().add_artist(circle3)

    Go1 = centro[0][0]
    So1 = centro[0][1]
    Go2 = centro[1][0]
    So2 = centro[1][1]
    Go3 = centro[2][0]
    So3 = centro[2][1]

    """create a matrix to see if a pixels is into the circle, using circle equation
    so the negative values of Mi means that the pixel belong to the circle"""
    M1 = (g - Go1) ** 2 + (s - So1) ** 2 - Ro ** 2
    M2 = (g - Go2) ** 2 + (s - So2) ** 2 - Ro ** 2
    M3 = (g - Go3) ** 2 + (s - So3) ** 2 - Ro ** 2

    aux1 = np.zeros([len(img_mean), len(img_mean[0])])
    aux2 = np.zeros([len(img_mean), len(img_mean[0])])
    aux3 = np.zeros([len(img_mean), len(img_mean[0])])
    for i in range(0, len(img_mean)):
        for j in range(0, len(img_mean[0])):
            if img_mean[i][j] > ic:
                if M1[i][j] < 0:
                    aux1[i][j] = 1
                if M2[i][j] < 0:
                    aux2[i][j] = 1
                if M3[i][j] < 0:
                    aux3[i][j] = 1

    img_new = np.copy(img_mean)

    indices1 = np.where(aux1 == 1)
    indices2 = np.where(aux2 == 1)
    indices3 = np.where(aux3 == 1)

    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))

    # Set the colors
    rgba[indices1[0], indices1[1], :3] = 1, 0, 0  # blue
    rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
    rgba[indices3[0], indices3[1], :3] = 0, 0, 1  # red

    plt.figure(2)
    plt.imshow(rgba)
    plt.title('Pseudocolor image')
    plt.show()

    return rgba
def lakeFiller(obs,start_coords):
    import numpy as np
    obsTemp = np.copy(obs)
    xsize,ysize = obs.shape
    stack = set(((start_coords[0], start_coords[1]),))
    orig_value = obs[start_coords[0],start_coords[1]]

    while stack:
        x, y = stack.pop()

        if obsTemp[x, y] == orig_value:
            obsTemp[x, y] = 1
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))

    return abs(obsTemp-1)+obs

def acessibleSpace(obs,start_coords):
    import numpy as np
    import matplotlib.pyplot as plt

    obsTemp = np.copy(obs)
    ysize,xsize = obs.shape
    stack = set(((start_coords[0], start_coords[1]),))
    orig_value = obs[start_coords[0],start_coords[1]]

    while stack:
        y, x = stack.pop()

        if obsTemp[y, x] == orig_value:
            obsTemp[y, x] = 1
            if x > 0:
                stack.add((y, x-1))
            if x < (xsize - 1):
                stack.add((y, x + 1))
            if y >= 0:
                if x!=(0):
                    stack.add(( (y - 1) % ysize , x))
            if y < (ysize):
                if x!=(0):
                    stack.add(( (y + 1) % ysize , x))

        # plt.figure(2)
        # fig = plt.figure(2)
        # plt.imshow(obsTemp-obs)
        # plt.pause(0.5)
        # plt.close(fig)

    return (obsTemp-obs)

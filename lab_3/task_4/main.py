if __name__ == "__main__":
    # X = [0.0, 0.1, 0.2, 0.3, 0.4]
    # Y = [1.0, 1.1052, 1.2214, 1.3499, 1.4918]
    # dest_X = 0.2

    X = [0.0, 1.0, 2.0, 3.0, 4.0]
    Y = [0.0, 0.5, 0.86603, 1.0, 0.86603]
    dest_X = 2.0

    for i in range(1, len(X)-1):
        if X[i-1] <= dest_X and dest_X <= X[i]:
            dy_left = (Y[i] - Y[i-1]) / (X[i] - X[i-1])
            dy_right = (Y[i+1] - Y[i]) / (X[i+1] - X[i])
            half_ddy = (dy_right - dy_left) / (X[i+1] - X[i-1])
            dy2 = dy_left + half_ddy * (2 * dest_X - X[i-1] - X[i])
            ddy = 2 * half_ddy
            
            print(f"{dy_left=}")
            print(f"{dy_right=}")
            print(f"{dy2=}")
            print(f"{ddy=}")
            break
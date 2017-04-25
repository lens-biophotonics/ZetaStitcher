import numpy as np

numbers = []
fill_value = 255

canvas = np.zeros((50, 30), dtype=np.uint8)
cx = canvas.shape[1] // 2
cy = canvas.shape[0] // 2

# 0
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[-2:, :] = fill_value
a[:, -2:] = fill_value
a[:, :2] = fill_value
numbers.append(a)

# 1
a = np.zeros_like(canvas)
a[:, cx - 1:cx + 1] = fill_value
numbers.append(a)

# 2
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[:cy,-2:] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[cy:, :2] = fill_value
a[-2:, :] = fill_value
numbers.append(a)

# 3
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[-2:, :] = fill_value
a[:, -2:] = fill_value
numbers.append(a)

# 4
a = np.zeros_like(canvas)
a[:cy, :2] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[:, -2:] = fill_value
numbers.append(a)

# 5
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[:cy,:2] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[cy:, -2:] = fill_value
a[-2:, :] = fill_value
numbers.append(a)

# 6
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[-2:, :] = fill_value
a[cy:, -2:] = fill_value
a[:, :2] = fill_value
numbers.append(a)

# 7
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[:, -2:] = fill_value
numbers.append(a)

# 8
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[-2:, :] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[:, -2:] = fill_value
a[:, :2] = fill_value
numbers.append(a)

# 9
a = np.zeros_like(canvas)
a[:2, :] = fill_value
a[-2:, :] = fill_value
a[cy - 1:cy + 1,:] = fill_value
a[:, -2:] = fill_value
a[:cy, :2] = fill_value
numbers.append(a)
